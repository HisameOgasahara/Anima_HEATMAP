import json
from pathlib import Path

import numpy as np
import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview

from anima_heatmap import CaptureConfig, CaptureSession, load_maps, render_map, save_views, select_indices
from .bridge import attach_capture, clean_conditioning, read_token_map
from .phrases import split_phrases, resolve_phrase, describe_selection
from .labels import label_images


class AnimaHeatmapTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "token_table")
    FUNCTION = "encode"
    CATEGORY = "Anima Heatmap"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        if "t5xxl" not in tokens or "qwen3_06b" not in tokens:
            raise ValueError("Anima CLIP을 연결하세요.")
        if len(tokens["t5xxl"]) != 1:
            raise ValueError("T5 토큰 스트림이 한 묶음이어야 합니다.")
        pairs = tokens["t5xxl"][0]
        decoded = clip.tokenizer.t5xxl.untokenize(pairs)
        pieces = [str(item[1]) for item in decoded]
        info = dict(prompt=text, ids=[int(pair[0]) for pair in pairs], pieces=pieces,
                    special_indices=[i for i, piece in enumerate(pieces) if piece in ("<pad>", "</s>", "<unk>")])
        encoded = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        condition = encoded.pop("cond")
        encoded["anima_heatmap_tokens"] = info
        table = "\n".join(f"{i}: {token} {piece!r}" for i, (token, piece) in enumerate(zip(info["ids"], pieces)))
        return ([[condition, encoded]], table)


class AnimaHeatmapSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "relations": (["image->text", "image->image", "both"],),
            "steps": ("STRING", {"default": "all", "tooltip": "all / last / 0,4,8 / 0-9 (0부터 시작)"}),
            "layers": ("STRING", {"default": "all"}),
            "heads": ("STRING", {"default": "mean", "tooltip": "mean / all / 0,2,4"}),
            "branches": (["positive", "negative", "both"],),
            "image_queries": ("STRING", {"default": "center", "tooltip": "center / all / sample:9 / 0,100"}),
            "text_keys": ("STRING", {"default": "all"}),
            "query_chunk": ("INT", {"default": 128, "min": 1, "max": 4096}),
            "key_chunk": ("INT", {"default": 256, "min": 1, "max": 4096}),
            "max_capture_gib": ("FLOAT", {"default": 8.0, "min": 0.01, "max": 1024.0}),
        }}

    RETURN_TYPES = ("ANIMA_HEATMAP_SETTINGS",)
    FUNCTION = "configure"
    CATEGORY = "Anima Heatmap"

    def configure(self, relations, steps, layers, heads, branches, image_queries, text_keys,
                  query_chunk, key_chunk, max_capture_gib):
        return (CaptureConfig(relations=("image->text", "image->image") if relations == "both" else (relations,),
            steps=steps, layers=layers, heads=heads,
            branches=("positive", "negative") if branches == "both" else (branches,),
            image_queries=image_queries, text_keys=text_keys, query_chunk=query_chunk, key_chunk=key_chunk,
            max_capture_bytes=int(max_capture_gib * 1024**3)),)


class AnimaHeatmapSampler:
    @classmethod
    def INPUT_TYPES(cls):
        from nodes import KSampler
        inputs = KSampler.INPUT_TYPES()
        inputs["required"]["settings"] = ("ANIMA_HEATMAP_SETTINGS",)
        return inputs

    RETURN_TYPES = ("LATENT", "ANIMA_HEATMAP_SESSION", "STRING")
    RETURN_NAMES = ("latent", "session", "session_directory")
    FUNCTION = "sample"
    CATEGORY = "Anima Heatmap"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, settings, denoise=1.0):
        token_maps = {"positive": read_token_map(positive), "negative": read_token_map(negative)}
        root = Path(folder_paths.get_output_directory()) / "anima_heatmap"
        with CaptureSession(root, settings, token_maps=token_maps,
                metadata=dict(seed=seed, steps=steps, cfg=cfg, sampler=sampler_name,
                              scheduler=scheduler, denoise=denoise, torch_version=torch.__version__)) as session:
            schedule = comfy.samplers.KSampler(model, steps=steps, device=model.load_device,
                sampler=sampler_name, scheduler=scheduler, denoise=denoise,
                model_options=model.model_options).sigmas
            with attach_capture(model, session, schedule.tolist()) as (patched, state):
                latent = comfy.sample.fix_empty_latent_channels(patched, latent_image["samples"],
                    latent_image.get("downscale_ratio_spacial"), latent_image.get("downscale_ratio_temporal"))
                noise = comfy.sample.prepare_noise(latent, seed, latent_image.get("batch_index"))
                preview = latent_preview.prepare_callback(patched, steps)

                def callback(step, x0, x, total):
                    if preview:
                        preview(step, x0, x, total)

                samples = comfy.sample.sample(patched, noise, steps, cfg, sampler_name, scheduler,
                    clean_conditioning(positive), clean_conditioning(negative), latent,
                    denoise=denoise, sigmas=schedule, noise_mask=latent_image.get("noise_mask"), callback=callback,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
                if not session.records:
                    raise RuntimeError("attention이 수집되지 않았습니다. 모델/선택 범위/CFG를 확인하세요.")
        output = latent_image.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = samples
        return (output, str(session.path), str(session.path))


class AnimaHeatmapLoadSession:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("ANIMA_HEATMAP_SESSION", "STRING")
    FUNCTION = "load"
    CATEGORY = "Anima Heatmap"

    def load(self, directory):
        root = Path(directory).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise ValueError("완료된 세션을 선택하세요.")
        return (str(root), json.dumps(manifest, ensure_ascii=False, indent=2))


class AnimaHeatmapView:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "session": ("ANIMA_HEATMAP_SESSION",), "images": ("IMAGE",),
            "relation": (["image->text", "image->image"],),
            "branch": (["positive", "negative"],),
            "batch_index": ("INT", {"default": 0, "min": 0}),
            "phrase": ("STRING", {"default": "cat", "multiline": True, "tooltip": "프롬프트 표현 그대로 입력. 여러 단어/태그는 줄바꿈 또는 쉼표로 구분. 괄호·가중치는 자동 처리합니다."}),
            "token_indices": ("STRING", {"default": "", "tooltip": "고급/디버깅용 수동 위치 선택. 보통은 비워 두세요. 입력하면 phrase보다 우선합니다."}),
            "occurrence": ("STRING", {"default": "all", "tooltip": "동일 단어의 전체 출현 또는 0부터 시작하는 출현 번호"}),
            "query_index": ("INT", {"default": -1, "min": -1, "tooltip": "-1이면 저장된 첫 이미지 query"}),
            "steps": ("STRING", {"default": "all"}), "layers": ("STRING", {"default": "all"}),
            "heads": ("STRING", {"default": "all"}), "calls": ("STRING", {"default": "all"}),
            "view": (["aggregate", "per_step", "per_layer", "per_record"],),
            "aggregation": (["mean", "daam"],),
            "normalization": (["relative", "shared", "absolute"],),
            "colormap": (["turbo", "viridis", "inferno", "magma", "gray"],),
            "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            "save_files": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("overlays", "heatmaps", "details")
    FUNCTION = "view"
    CATEGORY = "Anima Heatmap"

    def view(self, session, images, relation, branch, batch_index, phrase, token_indices,
             occurrence, query_index, steps, layers, heads, calls, view, aggregation,
             normalization, colormap, alpha, save_files):
        manifest = json.loads((Path(session) / "manifest.json").read_text(encoding="utf-8"))
        token_map = manifest["token_maps"][branch]
        selections, info = [], []
        if relation == "image->image":
            selections = [("", None)]
        elif token_indices.strip():
            ids = select_indices(token_indices, len(token_map["ids"]))
            selections = [(f"tokens {ids}", ids)]
            info.append(describe_selection(token_map, phrase, ids, "수동 위치 선택"))
        else:
            for query in split_phrases(phrase):
                ids, matched = resolve_phrase(token_map, query, occurrence)
                selections.append((query, ids))
                info.append(describe_selection(token_map, query, ids, matched))
            if not selections:
                raise ValueError("phrase에 프롬프트의 단어 또는 태그를 입력하세요.")
        items = []
        for query, ids in selections:
            items.extend(load_maps(session, relation=relation, branch=branch, batch=batch_index,
                phrase=query, token_indices=ids, occurrence=occurrence,
                query_index=None if query_index == -1 else query_index,
                steps=steps, layers=layers, heads=heads, calls=calls, view=view, aggregation=aggregation))
        if batch_index >= images.shape[0]:
            raise ValueError("이미지 batch 번호가 범위를 벗어났습니다.")
        base = images[batch_index].detach().cpu().float().numpy()
        value_range = (min(float(x["raw"].min()) for x in items), max(float(x["raw"].max()) for x in items))
        rendered = [render_map(item, image=base, alpha=alpha, colormap=colormap,
                    normalization=normalization, value_range=value_range) for item in items]
        info.extend(f"출력 {i}: {item['label']}" for i, item in enumerate(items))
        if save_files:
            saved = save_views(items, Path(session) / "views", image=base, alpha=alpha,
                               colormap=colormap, normalization=normalization)
            info.append(f"저장: {saved}")
        labels = [item["label"] for item in items]
        return (torch.from_numpy(label_images([r["overlay"] for r in rendered], labels)),
                torch.from_numpy(label_images([r["heatmap"] for r in rendered], labels)), "\n".join(info))


NODE_CLASS_MAPPINGS = {
    "AnimaHeatmapTextEncode": AnimaHeatmapTextEncode,
    "AnimaHeatmapSettings": AnimaHeatmapSettings,
    "AnimaHeatmapSampler": AnimaHeatmapSampler,
    "AnimaHeatmapLoadSession": AnimaHeatmapLoadSession,
    "AnimaHeatmapView": AnimaHeatmapView,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaHeatmapTextEncode": "Anima Heatmap · Text Encode",
    "AnimaHeatmapSettings": "Anima Heatmap · Settings",
    "AnimaHeatmapSampler": "Anima Heatmap · KSampler",
    "AnimaHeatmapLoadSession": "Anima Heatmap · Load Session",
    "AnimaHeatmapView": "Anima Heatmap · View",
}
