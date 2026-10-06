from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .influence import AnimaTagInfluenceSampler, AnimaTagInfluenceView
from .v_ablation import AnimaVAblationSampler

NODE_CLASS_MAPPINGS.update(AnimaTagInfluenceSampler=AnimaTagInfluenceSampler,
                         AnimaTagInfluenceView=AnimaTagInfluenceView,
                         AnimaVAblationSampler=AnimaVAblationSampler)
NODE_DISPLAY_NAME_MAPPINGS.update(AnimaTagInfluenceSampler="Anima Heatmap · Tag Influence Sampler",
                                AnimaTagInfluenceView="Anima Heatmap · Tag Influence View",
                                AnimaVAblationSampler="Anima Heatmap · V Ablation Sampler")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

