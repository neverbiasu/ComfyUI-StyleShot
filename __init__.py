from .node import PipelineLoader, StyleShotApply

NODE_CLASS_MAPPINGS = {
    "PipelineLoader": PipelineLoader,
    "StyleShotApply": StyleShotApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipelineLoader": "Pipeline Loader",
    "StyleShotApply": "StyleShotApply",
}
