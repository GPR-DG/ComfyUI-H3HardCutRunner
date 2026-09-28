# ComfyUI-H3HardCutRunner

This plugin contains five independent MiniMax H3 hard-cut runner nodes plus a
shot-local scene-analysis helper:

1. `H3HardCutRunner` — the original formal hard-cut runner.
2. `H3HardCutRunnerShotPromptEmptyShot` — validation candidate A with
   shot-local scene prompts, an empty-shot policy, and Depth as the H3
   reference video.
3. `H3HardCutRunnerFunControl` — validation candidate B with shot-local scene
   prompts, an empty-shot policy, and MiniMax H3 Fun Control Depth.
4. `H3HardCutRunnerSceneVLM` — validation candidate C. It preserves A's
   generation sequence, adds optional `picture3`, and consumes strict JSONL
   prompts from `H3ShotSceneVLM`.
5. `H3HardCutRunnerSceneVLMFunControl` — validation candidate D. It keeps the
   same shot-local scene/VLM contract but transports structural control through
   the official MiniMax H3 Fun ControlNet model patch instead of H3
   `ref_video_0`.

`H3ShotSceneVLM` uses the same deterministic hard-cut detector as the runners,
creates one shared installed Qwen-VL node instance per run, invokes it once
per detected shot, and emits exactly one validated scene record per shot plus
dynamic empty-shot indices. It does not receive the target character
reference image, so scene analysis cannot silently ingest Picture1/Picture2
identity or white-background content. Its report retains the normalized scene
fields and separates source-protagonist presence from secondary/background human
signals for diagnosis; only the source-protagonist field controls reference
injection.

Picture3 is a capability-only optional back-garment reference. When it is not
connected, Runner C omits the third H3 reference entirely; it is not a root-
cause fix for garment, prop, or accessory drift.

Complex optional Picture3 wording in production prompts must be wrapped in
`[[PICTURE3_CONTRACT_BEGIN]]` and `[[PICTURE3_CONTRACT_END]]`. Runner C/D keeps
the complete block when Picture3 is connected and removes the complete block
when it is absent; unmarked legacy prose is cleaned conservatively.

Candidate B additionally requires `MiniMaxH3FunControlNetApply`,
`ModelPatchLoader`, and the model patch
`minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors` to exist in
the target ComfyUI environment. RunningHub support for those dependencies is
not asserted here; the runtime requires them to be installed and registered.

Candidate D is an evidence-backed root-cause candidate for source-structure
leakage observed in the current A sample: source Depth can carry coarse
garment and handheld-prop silhouettes when it is transported as H3 reference
video. D omits H3 `ref_video_0` and applies Depth as `control_video` through
Fun ControlNet. This is not visual proof of a fix until the same RunningHub
environment produces a C/D comparison. D also requires the official Fun
ControlNet node and patch model; no RunningHub availability is assumed.

A, B, C, and D are validation candidates, not proven final-production
runners. Keep every node type as a separate module so they can be tested
independently.
Install this directory under `ComfyUI/custom_nodes/` and restart ComfyUI.

