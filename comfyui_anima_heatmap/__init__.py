from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .influence import AnimaTagInfluenceSampler, AnimaTagInfluenceView

NODE_CLASS_MAPPINGS.update(AnimaTagInfluenceSampler=AnimaTagInfluenceSampler,
                         AnimaTagInfluenceView=AnimaTagInfluenceView)
NODE_DISPLAY_NAME_MAPPINGS.update(AnimaTagInfluenceSampler="Anima Heatmap · Tag Influence Sampler",
                                AnimaTagInfluenceView="Anima Heatmap · Tag Influence View")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

