# ComfyUI-H3HardCutRunner

This plugin retains five independent MiniMax H3 hard-cut runner nodes and a
shot-local scene-analysis helper for compatibility. The formal C workflow now
uses the modular C nodes described below instead of the legacy C runner.

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

Picture3 is a capability-only optional back-garment reference. In the modular
C workflow the upload node is connected to the policy node; no selected image
produces `None`, which the H3 reference node omits. Picture3 is not a root-
cause fix for garment, prop, or accessory drift.

## Modular C workflow

`H3CShotPlanner`, `H3CShotSelectPad`, `H3CShotPolicy`, and
`H3CPromptCompiler` expose the shot manifest, aligned RGB/Depth, per-shot
reference decisions, and both final prompts. `H3CShotTrim`,
`H3COrderedMergeStep`, and `H3COrderedMergeFinish` restore original shot
lengths and enforce ordered, full-length output. The canvas uses Easy-Use's
`easy forLoopStart` / `easy forLoopEnd` and visible Comfy/RH H3 conditioning,
noise, guider, two-pass sampling, AV split/concat, latent upscaling, VAE decode,
CreateVideo, and SaveVideo nodes. The VLM node appends raw and normalized
diagnostic outputs without changing its first three outputs.

Static contract tests are not RunningHub execution proof. Verify the installed
Easy-Use loop version, H3 autogrow `ref_image_2`/`ref_video_0`, and a multi-shot
run on the target RH instance before treating this workflow as deployed. The
Shot1/9 appearance reversions, Shot2/8 waist leakage, Shot5 sleeve transfer,
Shot7 secondary-person contamination, Depth-reference competition, VLM
semantic leakage, whole-image Picture2 reference, first-pass structure lock,
and 39-frame quality risk remain open for RH A/B diagnosis. The modularization
does not claim to fix any of them.

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
runners. Legacy runner variants have separate modules; the seven C Modular
classes share one Python module but register as independent canvas nodes.
Install this directory under `ComfyUI/custom_nodes/` and restart ComfyUI.

## Repository layout and C Modular checks

```text
ComfyUI-H3HardCutRunner/
├── __init__.py
├── h3_hardcut_runner_a.py
├── h3_hardcut_runner_b.py
├── h3_hardcut_runner_c.py
├── h3_hardcut_runtime.py
├── h3_hardcut_modular_c.py
├── tools/
│   └── build_h3_modular_c.py
└── tests/
    └── test_h3_modular_c.py
```

The seven C Modular node classes intentionally remain in one Python module;
`NODE_CLASS_MAPPINGS` registers each as a separate canvas node.

From the repository root, run `python tools/build_h3_modular_c.py` and
`python -m unittest discover -s tests -p 'test_h3_modular_c.py'`. The builder
requires the immutable C baseline under the surrounding project directory's
`work/modular_c_baseline_20261001/` and writes the formal C JSON under its
`outputs/`. When the repository is checked out elsewhere, set
`H3_C_PROJECT_ROOT` to the project directory containing those two folders.
The baseline and generated workflow are deliberately not part of this plugin
repository. Building and unit tests do not execute RunningHub generation.

For the diagnostic A2 copy only, run
`python tools/build_h3_modular_c.py --a2-force-refs`. It reads the current
formal C JSON without overwriting it and writes
`outputs/H3_V16_hardcut_runner_RH_SCENE_VLM_C_A2_FORCE_REFS.json` with
`manual_force_present_indices=1,9` on `H3CShotPolicy`. The formal C builder
emits the same optional input with a blank default; A2 narrows the diagnostic
comparison to the two whole-outfit reversion shots.
An explicit force-empty/force-present overlap fails closed; the override
does not validate or fix other sources of appearance or geometry drift.
