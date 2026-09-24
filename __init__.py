from .h3_hardcut_runtime import H3HardCutRunner
from .h3_hardcut_runner_a import H3HardCutRunnerShotPromptEmptyShot
from .h3_hardcut_runner_b import H3HardCutRunnerFunControl

NODE_CLASS_MAPPINGS = {
    "H3HardCutRunner": H3HardCutRunner,
    "H3HardCutRunnerShotPromptEmptyShot": H3HardCutRunnerShotPromptEmptyShot,
    "H3HardCutRunnerFunControl": H3HardCutRunnerFunControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3HardCutRunner": "H3 Hard-Cut Runner (6+2)",
    "H3HardCutRunnerShotPromptEmptyShot": "H3 Hard-Cut Runner A (Shot Prompt + Empty Shot)",
    "H3HardCutRunnerFunControl": "H3 Hard-Cut Runner B (Fun Control + Shot-local Depth)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
