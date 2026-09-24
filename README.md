# ComfyUI-H3HardCutRunner

This plugin contains three independent MiniMax H3 hard-cut runner nodes:

1. `H3HardCutRunner` — the original formal hard-cut runner.
2. `H3HardCutRunnerShotPromptEmptyShot` — validation candidate A with
   shot-local scene prompts, an empty-shot policy, and Depth as the H3
   reference video.
3. `H3HardCutRunnerFunControl` — validation candidate B with shot-local scene
   prompts, an empty-shot policy, and MiniMax H3 Fun Control Depth.

Candidate B additionally requires `MiniMaxH3FunControlNetApply`,
`ModelPatchLoader`, and the model patch
`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` to exist in
the target ComfyUI environment. RunningHub support for those dependencies is
not asserted here; the runtime requires them to be installed and registered.

A and B are validation candidates, not proven final-production runners. Keep
the three node types as separate modules so they can be tested independently.
Install this directory under `ComfyUI/custom_nodes/` and restart ComfyUI.
