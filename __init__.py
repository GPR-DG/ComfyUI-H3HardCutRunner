from .h3_hardcut_runtime import H3HardCutRunner
from .h3_hardcut_runner_a import H3HardCutRunnerShotPromptEmptyShot
from .h3_hardcut_runner_b import H3HardCutRunnerFunControl
from .h3_hardcut_runner_c import (
    H3ShotSceneVLM,
    H3OptionalPicture3,
    H3HardCutRunnerSceneVLM,
    H3HardCutRunnerSceneVLMFunControl,
)
from .h3_hardcut_modular_c import (
    NODE_CLASS_MAPPINGS as C_MODULAR_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as C_MODULAR_NODE_DISPLAY_NAME_MAPPINGS,
)
from .h3_hardcut_ref_adapters import (
    NODE_CLASS_MAPPINGS as C_REF_ADAPTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as C_REF_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .h3_shot_reference_router import (
    NODE_CLASS_MAPPINGS as C_REFERENCE_ROUTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as C_REFERENCE_ROUTER_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS = {
    "H3HardCutRunner": H3HardCutRunner,
    "H3HardCutRunnerShotPromptEmptyShot": H3HardCutRunnerShotPromptEmptyShot,
    "H3HardCutRunnerFunControl": H3HardCutRunnerFunControl,
    "H3ShotSceneVLM": H3ShotSceneVLM,
    "H3OptionalPicture3": H3OptionalPicture3,
    "H3HardCutRunnerSceneVLM": H3HardCutRunnerSceneVLM,
    "H3HardCutRunnerSceneVLMFunControl": H3HardCutRunnerSceneVLMFunControl,
}
NODE_CLASS_MAPPINGS.update(C_MODULAR_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(C_REF_ADAPTER_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(C_REFERENCE_ROUTER_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3HardCutRunner": "H3 Hard-Cut Runner (6+2)",
    "H3HardCutRunnerShotPromptEmptyShot": "H3 Hard-Cut Runner A (Shot Prompt + Empty Shot)",
    "H3HardCutRunnerFunControl": "H3 Hard-Cut Runner B (Fun Control + Shot-local Depth)",
    "H3ShotSceneVLM": "H3 Shot Scene VLM (Per-Shot, Dynamic)",
    "H3OptionalPicture3": "H3 Optional Picture3 Upload",
    "H3HardCutRunnerSceneVLM": "H3 Hard-Cut Runner C (Scene VLM + Picture3)",
    "H3HardCutRunnerSceneVLMFunControl": "H3 Hard-Cut Runner D (Scene VLM + Fun ControlNet)",
}
NODE_DISPLAY_NAME_MAPPINGS.update(C_MODULAR_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(C_REF_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(C_REFERENCE_ROUTER_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
# Background nodes are optional at import time so an existing installation keeps
# its original nodes visible when NumPy/OpenCV are not installed yet.
try:
    from .h3_background_nodes import (
        NODE_CLASS_MAPPINGS as BACKGROUND_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as BACKGROUND_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception:
    import logging
    logging.getLogger(__name__).warning(
        "H3 background nodes unavailable; existing H3 nodes remain registered.",
        exc_info=True,
    )
    BACKGROUND_NODE_CLASS_MAPPINGS = {}
    BACKGROUND_NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(BACKGROUND_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(BACKGROUND_NODE_DISPLAY_NAME_MAPPINGS)
