"""ComfyUI의 Anima 실행을 독립 모듈의 입력 규약에 연결한다."""

from contextlib import contextmanager
import math
import threading
from types import MethodType

from anima_heatmap import select_indices

_HOOK_LOCK = threading.RLock()


def read_token_map(conditioning):
    if len(conditioning) != 1:
        raise ValueError("텍스트 위치를 보존하려면 분기마다 Text Encode 노드 한 개를 연결하세요.")
    info = conditioning[0][1]
    if "anima_heatmap_tokens" not in info:
        raise ValueError("Anima Heatmap Text Encode로 positive와 negative를 인코딩하세요.")
    if any(key in info for key in ("area", "mask", "start_percent", "end_percent", "control")):
        raise ValueError("영역/시간 조건 및 ControlNet은 아직 이 수집 연결부에서 지원하지 않습니다.")
    return info["anima_heatmap_tokens"]


def clean_conditioning(conditioning):
    return [[tensor, {key: value for key, value in info.items() if key != "anima_heatmap_tokens"}]
            for tensor, info in conditioning]


@contextmanager
def attach_capture(model, session, sigmas):
    """이 모델 인스턴스의 attention 입력 Q/K를 관찰하고 종료 시 원상 복구한다."""
    from comfy.ldm.anima.model import Anima

    diffusion = model.get_model_object("diffusion_model")
    if not isinstance(diffusion, Anima):
        raise TypeError("ComfyUI 기본 Anima 모델을 연결하세요.")
    if getattr(diffusion, "_orig_mod", None) is not None:
        raise ValueError("수집할 때는 torch.compile을 적용하지 않은 Anima를 사용하세요.")
    state = {"call": 0}
    schedule = [float(s) for s in sigmas]
    total_steps = len(schedule) - 1
    if total_steps < 1 or any(a < b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("내림차순 noise schedule이 필요합니다.")
    session.manifest["metadata"]["sigmas"] = schedule
    session.manifest["metadata"]["step_semantics"] = "noise schedule interval, not solver callback count"
    patched = model.clone()
    previous_wrapper = patched.model_options.get("model_function_wrapper")
    restored = []
    marker = str(session.path)
    total_layers = len(diffusion.blocks)
    selected_layers = select_indices(session.config.layers, total_layers)
    select_indices(session.config.steps, total_steps)

    def model_wrapper(apply_model, args):
        labels = args.get("cond_or_uncond")
        if not labels or any(label not in (0, 1) for label in labels):
            raise ValueError("긍정/부정 분기 정보를 확인할 수 없습니다.")
        c = args["c"].copy()
        options = c.get("transformer_options", {}).copy()
        if "optimized_attention_override" in options:
            raise ValueError("attention 연산을 교체한 모델은 기본 Q/K 확률과 달라질 수 있어 수집할 수 없습니다.")
        shape = args["input"].shape
        if len(shape) == 4:
            t, h, w = 1, shape[-2], shape[-1]
        else:
            t, h, w = shape[-3:]
        sigma = float(args["timestep"].max().detach().cpu())
        step = next((i for i, lower in enumerate(schedule[1:])
                     if sigma > lower and not math.isclose(sigma, lower, rel_tol=1e-6, abs_tol=1e-8)),
                    total_steps - 1)
        options["anima_heatmap_capture"] = dict(session=marker, labels=list(labels),
            step=step, call=state["call"], sigma=sigma,
            grid=(math.ceil(t / diffusion.patch_temporal), math.ceil(h / diffusion.patch_spatial), math.ceil(w / diffusion.patch_spatial)),
            valid_grid=(t / diffusion.patch_temporal, h / diffusion.patch_spatial, w / diffusion.patch_spatial))
        state["call"] += 1
        c["transformer_options"] = options
        if previous_wrapper is not None:
            return previous_wrapper(apply_model, args | {"c": c})
        return apply_model(args["input"], args["timestep"], **c)

    def make_hook(original, layer, relation):
        def observe(module, q, k, v, transformer_options=None):
            options = transformer_options or {}
            info = options.get("anima_heatmap_capture", {})
            if info.get("session") != marker:
                return original(q, k, v, transformer_options=options)
            if not session.wants(relation, info["step"], layer, total_steps, total_layers):
                return original(q, k, v, transformer_options=options)
            spatial = info["grid"]
            if math.prod(q.shape[1:-2]) != math.prod(spatial) or spatial[0] != 1:
                raise ValueError(f"Anima 이미지 Q 격자가 예상과 다릅니다: {tuple(q.shape)}")
            labels = info["labels"]
            if q.shape[0] % len(labels):
                raise ValueError("attention batch와 분기 수가 일치하지 않습니다.")
            if len(set(labels)) != len(labels):
                raise ValueError("한 번의 호출에 동일 분기의 여러 조건이 포함되어 있습니다.")
            batch_size = q.shape[0] // len(labels)
            for index, label in enumerate(labels):
                branch = "positive" if label == 0 else "negative"
                if branch not in session.config.branches:
                    continue
                part = slice(index * batch_size, (index + 1) * batch_size)
                qs = q[part].reshape(batch_size, -1, q.shape[-2], q.shape[-1]).transpose(1, 2)
                ks = k[part].reshape(batch_size, -1, k.shape[-2], k.shape[-1]).transpose(1, 2)
                session.capture(qs, ks, relation=relation, branch=branch,
                    step=info["step"], call=info["call"], layer=layer, grid=spatial,
                    valid_grid=info["valid_grid"], total_steps=total_steps,
                    total_layers=total_layers, sigma=info["sigma"])
            return original(q, k, v, transformer_options=options)
        return observe

    with _HOOK_LOCK:
        try:
            for layer in selected_layers:
                block = diffusion.blocks[layer]
                for relation, module in (("image->text", block.cross_attn), ("image->image", block.self_attn)):
                    if relation not in session.config.relations:
                        continue
                    had_instance_method = "compute_attention" in module.__dict__
                    saved = module.__dict__.get("compute_attention")
                    original = module.compute_attention
                    restored.append((module, had_instance_method, saved))
                    module.compute_attention = MethodType(make_hook(original, layer, relation), module)
            patched.set_model_unet_function_wrapper(model_wrapper)
            yield patched, state
        finally:
            for module, existed, saved in reversed(restored):
                if existed:
                    module.compute_attention = saved
                else:
                    del module.compute_attention

