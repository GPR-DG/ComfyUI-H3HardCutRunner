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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
