"""동일 latent와 sigma에서 조건을 바꾼 denoised 예측 차이를 수집한다."""

import math

import numpy as np
import torch

import comfy.model_management
import comfy.sample
import comfy.samplers


class InfluenceGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, positive, negative, ablated, cfg, sigmas=None):
        super().__init__(model)
        self.ablations = ablated if isinstance(ablated, dict) else {"ablated": ablated}
        self.inner_set_conds(dict(positive=positive, negative=negative,
            **{f"ablated_{name}": cond for name, cond in self.ablations.items()}))
        self.set_cfg(cfg)
        self.sigmas = list(sigmas) if sigmas is not None else None
        self.targets = {name: dict(records=[], denoised_maps=[], velocity_maps=[])
            for name in self.ablations}
        first = next(iter(self.targets.values()))
        self.records = first["records"]
        self.denoised_maps = first["denoised_maps"]
        self.velocity_maps = first["velocity_maps"]
        self.baseline_velocity_maps = []

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        options = model_options or {}
        unsupported = ("sampler_cfg_function", "sampler_pre_cfg_function",
                       "sampler_post_cfg_function", "sampler_calc_cond_batch_function")
        if any(options.get(key) for key in unsupported):
            raise ValueError("태그 영향 측정은 기본 CFG 연산을 사용하세요.")
        negative = self.conds["negative"] if self.cfg != 1 else None
        positive, uncond = comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds["positive"], negative], x, timestep, options)
        original_prediction = positive if self.cfg == 1 else uncond + (positive - uncond) * self.cfg
        # 단독 호출 기준은 batch 수치 차이를 기록하는 진단에만 사용한다.
        # 태그 영향은 10/4 실험처럼 원래 긍정·부정 호출의 CFG 예측을 기준으로 한다.
        baseline_positive = positive if self.cfg == 1 else comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds["positive"]], x, timestep, options)[0]
        baseline_prediction = baseline_positive if self.cfg == 1 else uncond + (baseline_positive - uncond) * self.cfg
        # Flow 모델의 denoised = x - sigma * velocity. sigma 차이를 제거한다.
        sigma = timestep.reshape(-1, *([1] * (x.ndim - 1))).float()
        reference_velocity = (x.float() - original_prediction.float()) / sigma
        sigma_value = float(timestep[0])
        step = len(self.records)
        if self.sigmas is not None:
            # 기존 attention 수집과 동일하게 noise schedule 구간으로 묶는다.
            step = next((i for i, lower in enumerate(self.sigmas[1:])
                         if sigma_value > lower and not math.isclose(
                             sigma_value, lower, rel_tol=1e-6, abs_tol=1e-8)),
                        len(self.sigmas) - 2)
        baseline_map = torch.linalg.vector_norm(
            (original_prediction.float() - baseline_prediction.float()) / sigma, dim=1)
        self.baseline_velocity_maps.append(baseline_map.detach().cpu().numpy())
        for name, target in self.targets.items():
            ablated = comfy.samplers.calc_cond_batch(self.inner_model,
                [self.conds[f"ablated_{name}"]], x, timestep, options)[0]
            ablated_prediction = ablated if self.cfg == 1 else uncond + (ablated - uncond) * self.cfg
            delta = original_prediction.float() - ablated_prediction.float()
            velocity_delta = delta / sigma
            denoised_map = torch.linalg.vector_norm(delta, dim=1)
            velocity_map = torch.linalg.vector_norm(velocity_delta, dim=1)
            relative_rms = velocity_delta.square().mean().sqrt() / reference_velocity.square().mean().sqrt().clamp_min(1e-12)
            target["denoised_maps"].append(denoised_map.detach().cpu().numpy())
            target["velocity_maps"].append(velocity_map.detach().cpu().numpy())
            target["records"].append(dict(call=len(target["records"]), step=step, sigma=sigma_value,
                mean_velocity_l2=float(velocity_map.mean()), max_velocity_l2=float(velocity_map.max()),
                relative_velocity_rms=float(relative_rms),
                mean_batch_baseline_velocity_l2=float(baseline_map.mean())))
        # 비교용 예측은 sampler에 전달하지 않는다. 원래 조건의 경로만 진행한다.
        return original_prediction


def sample_influence(model, positive, negative, ablated, latent_image, *, seed,
                     steps, cfg, sampler_name, scheduler, denoise=1.0):
    import comfy.utils
    import latent_preview

    latent = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"],
        latent_image.get("downscale_ratio_spacial"))
    noise = comfy.sample.prepare_noise(latent, seed, latent_image.get("batch_index"))
    schedule = comfy.samplers.KSampler(model, steps=steps, device=model.load_device,
        sampler=sampler_name, scheduler=scheduler, denoise=denoise,
        model_options=model.model_options).sigmas
    guider = InfluenceGuider(model, positive, negative, ablated, cfg, schedule.tolist())
    preview = latent_preview.prepare_callback(model, len(schedule) - 1)
    with torch.inference_mode():
        samples = guider.sample(noise, latent, comfy.samplers.sampler_object(sampler_name),
            schedule, denoise_mask=latent_image.get("noise_mask"), callback=preview,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    output = latent_image.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples.to(device=comfy.model_management.intermediate_device(), dtype=torch.float32)
    return output, dict(records=guider.records, sigmas=schedule.tolist(),
        configured_steps=steps, total_steps=len(schedule) - 1,
        step_semantics="noise schedule interval, not model-call count",
        comparison="2026-10-04 legacy: original positive/negative CFG versus positive-only ablation; original CFG advances sampling",
        baseline_velocity_maps=np.stack(guider.baseline_velocity_maps),
        denoised_maps=np.stack(guider.denoised_maps), velocity_maps=np.stack(guider.velocity_maps),
        targets={name: dict(records=target["records"],
            denoised_maps=np.stack(target["denoised_maps"]), velocity_maps=np.stack(target["velocity_maps"]))
            for name, target in guider.targets.items()})


class AnimaTagInfluenceSampler:
    @classmethod
    def INPUT_TYPES(cls):
        from nodes import KSampler
        inputs = KSampler.INPUT_TYPES()
        inputs["required"]["ablated_positive"] = ("CONDITIONING",)
        return inputs

    RETURN_TYPES = ("LATENT", "ANIMA_TAG_INFLUENCE")
    RETURN_NAMES = ("latent", "influence")
    FUNCTION = "sample"
    CATEGORY = "Anima Heatmap"

    def sample(self, model, positive, negative, ablated_positive, latent_image,
               seed, steps, cfg, sampler_name, scheduler, denoise=1.0):
        return sample_influence(model, positive, negative, ablated_positive, latent_image,
            seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, denoise=denoise)


class AnimaTagInfluenceView:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"influence": ("ANIMA_TAG_INFLUENCE",), "images": ("IMAGE",),
            "scale_max": ("FLOAT", {"default": 0.0, "min": 0.0,
                "tooltip": "0이면 전체 측정 지도의 공통 최대값을 사용합니다."}),
            "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0})},
            "optional": {
                "view": (["aggregate", "per_step"],),
                "steps": ("STRING", {"default": "all", "tooltip": "all / last / 0,4,8 / 0-9 (0부터 시작)"}),
            }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("overlays", "heatmaps", "details")
    FUNCTION = "view"
    CATEGORY = "Anima Heatmap"

    def view(self, influence, images, scale_max, alpha, view="aggregate", steps="all"):
        import json
        from anima_heatmap import select_indices
        from anima_heatmap.maps import render_map
        from .labels import label_images

        records = influence["records"]
        maps = influence["velocity_maps"]
        total_steps = influence.get("total_steps", len(records))
        selected = select_indices(steps, total_steps)
        chosen = [i for i, record in enumerate(records)
                  if record.get("step", record["call"]) in selected]
        if not chosen:
            raise ValueError("선택한 단계의 영향 측정이 없습니다.")
        if view not in ("aggregate", "per_step"):
            raise ValueError("view는 aggregate 또는 per_step입니다.")
        groups = {"aggregate": chosen} if view == "aggregate" else {
            step: [i for i in chosen if records[i].get("step", records[i]["call"]) == step]
            for step in sorted({records[i].get("step", records[i]["call"]) for i in chosen})}
        # 단계 선택을 바꿔도 색 눈금은 전체 측정에 고정한다.
        maximum = scale_max if scale_max > 0 else max(float(maps.max()), 1e-12)
        results, labels = [], []
        for group, indices in groups.items():
            for index, raw in enumerate(maps[indices].mean(axis=0)):
                raw = raw.squeeze(0) if raw.ndim == 3 else raw
                grid = (1, *raw.shape)
                image = images[index].detach().cpu().float().numpy()
                results.append(render_map(dict(raw=raw, grid=grid, valid_grid=grid),
                    image=image, normalization="shared",
                    value_range=(0, maximum), colormap="turbo", alpha=alpha))
                name = "selected-step mean" if group == "aggregate" else f"step {group + 1}/{total_steps}"
                labels.append(f"Tag influence | {name} | batch {index} | scale 0..{maximum:.4g}")
        details = dict(total_steps=total_steps, model_calls=len(records), view=view,
            selected_steps=selected, scale_max=maximum, outputs=labels,
            background="final decoded image", records=[records[i] for i in chosen])
        return (torch.from_numpy(label_images([r["overlay"] for r in results], labels)),
            torch.from_numpy(label_images([r["heatmap"] for r in results], labels)),
            json.dumps(details, ensure_ascii=False, indent=2))
