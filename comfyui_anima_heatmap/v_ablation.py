"""원본 텍스트의 선택 토큰 V를 차단한 예측 차이를 측정한다."""

from contextlib import contextmanager
from functools import partial
from types import MethodType

from .bridge import _HOOK_LOCK, clean_conditioning, read_token_map
from .influence import InfluenceGuider, sample_influence
from .phrases import resolve_phrase


@contextmanager
def ablate_token_values(diffusion, token_indices):
    """비교 실행 동안 모든 cross-attention 층의 선택 V만 0으로 만든다."""
    restored = []

    def make_hook(original):
        def compute(module, x, context=None, rope_emb=None, transformer_options=None):
            q, k, v = original(x, context=context, rope_emb=rope_emb,
                               transformer_options=transformer_options)
            if v.ndim != 4 or max(token_indices) >= v.shape[1]:
                raise ValueError("선택한 토큰 위치와 Anima cross-attention V 구조가 일치하지 않습니다.")
            values = v.clone()
            values[:, token_indices, :, :] = 0
            return q, k, values
        return compute

    with _HOOK_LOCK:
        try:
            for block in diffusion.blocks:
                module = block.cross_attn
                restored.append((module, "compute_qkv" in module.__dict__,
                                 module.__dict__.get("compute_qkv")))
                module.compute_qkv = MethodType(make_hook(module.compute_qkv), module)
            yield
        finally:
            for module, existed, saved in reversed(restored):
                if existed:
                    module.compute_qkv = saved
                else:
                    del module.compute_qkv


class VAblationGuider(InfluenceGuider):
    def __init__(self, *args, diffusion, token_indices, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = diffusion
        self.token_indices = token_indices

    def predict_ablation(self, name, x, timestep, options):
        with ablate_token_values(self.diffusion, self.token_indices):
            return super().predict_ablation(name, x, timestep, options)


class AnimaVAblationSampler:
    @classmethod
    def INPUT_TYPES(cls):
        from nodes import KSampler
        inputs = KSampler.INPUT_TYPES()
        inputs["required"]["phrase"] = ("STRING", {"default": "",
            "tooltip": "원본 긍정 프롬프트에서 V를 차단할 태그. Anima Heatmap Text Encode를 연결하세요."})
        return inputs

    RETURN_TYPES = ("LATENT", "ANIMA_TAG_INFLUENCE")
    RETURN_NAMES = ("latent", "influence")
    FUNCTION = "sample"
    CATEGORY = "Anima Heatmap"

    def sample(self, model, positive, negative, latent_image, phrase,
               seed, steps, cfg, sampler_name, scheduler, denoise=1.0):
        from comfy.ldm.anima.model import Anima

        diffusion = model.get_model_object("diffusion_model")
        if not isinstance(diffusion, Anima):
            raise TypeError("ComfyUI 기본 Anima 모델을 연결하세요.")
        if getattr(diffusion, "_orig_mod", None) is not None:
            raise ValueError("V 차단 측정에는 torch.compile을 적용하지 않은 Anima를 사용하세요.")
        token_map = read_token_map(positive)
        indices, matched = resolve_phrase(token_map, phrase)
        factory = partial(VAblationGuider, diffusion=diffusion, token_indices=indices)
        original = clean_conditioning(positive)
        latent, influence = sample_influence(model, original, clean_conditioning(negative),
            original, latent_image, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler, denoise=denoise,
            guider_factory=factory)
        influence.update(method="v_ablation", phrase=phrase, matched_phrase=matched,
            token_indices=indices, layers="all cross-attention", heads="all",
            comparison="original positive versus same positive with selected token V zeroed; original CFG advances sampling")
        return latent, influence
