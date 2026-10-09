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

## Modular-C REF2VA / SelfLift Adapters

This plugin adds four Modular-C nodes for REF2VA / SelfLift shot preparation:

- `H3CShotDualRefPad`
- `H3CShotGuideWindowPlanner`
- `H3CShotGuideWindowSelect`
- `H3CShotGuideWindowAppend`

The adapter contract keeps RGB and Depth synchronized per shot, pads only with the current shot tail frame, and uses donor 124 / 102 / 22. AddGuide state resets after HardCut, with a 24fps upstream contract. H3 / AddGuide / SelfLift continue to use the existing nodes. RH / CUDA / GPU neural execution still requires real-device verification.

## Task-scoped background reference nodes

The audited background core is packaged in this plugin, with six thin nodes in
`video/MiniMaxH3/C Background`: `H3CBackgroundTaskStart`,
`H3CBestBackgroundFrame`, `H3CBackgroundResolve`, `H3CBackgroundWashClaim`,
`H3CBackgroundAttachClean`, and `H3CBackgroundTaskCleanup`.
Matching, Environment/View decisions, SQLite transactions, removal-mask evidence
and wash-token validation remain in the core; the wrapper does not reimplement them.

### Installation and dependencies

Clone `https://github.com/GPR-DG/ComfyUI-H3HardCutRunner` under
`ComfyUI/custom_nodes/`, or pull the chosen deployment branch in that existing
checkout, then restart ComfyUI. No separate copying of the three core directories
or manual node registration is needed. Existing H3 nodes still register if the
optional background import fails; the console includes its warning and traceback.

Use ComfyUI's existing Python and PyTorch. Background nodes also need NumPy and
an OpenCV build providing `cv2.SIFT_create`; SQLite is in Python's standard library.
The core's 124-test suite passes with OpenCV 4.13.0 / NumPy 2.2.6 and with
OpenCV 5.0.0 / NumPy 2.5.3. Older OpenCV versions are not certified by this suite;
having SIFT alone is not full behavioral qualification. These are tested pairs,
not permission to overwrite RunningHub's global dependencies. Check the existing
environment first; do not automatically upgrade/downgrade OpenCV or install both
`opencv-python` and `opencv-python-headless`. No global dependency changes are
required by a requirements file in this release.

### Wiring and lifecycle contract

Create **one TaskStart per video, outside the Shot and Window loops**. All Shots
share its task output. `task_id` is only a label: every execution appends a fresh
UUID, including runs with the same label or video filename. The default database
is task-local `:memory:`; an explicitly chosen disk database remains task-ID
isolated. Never use a fixed task identity to resume a different video.

Feed unpadded raw Shot RGB into BestBackgroundFrame once per Shot. Optional
`person_masks` must cover that Shot's frames (including secondary people); Depth
must align with it. Set `background_only=True` only with reliable explicit empty
scene evidence, never merely from an absent main subject. Missing masks with
`background_only=False` remain unknown, not an inferred empty scene.

Resolve returns a decision and a View. A PENDING View has `ready=False` and
`clean_background=None`: branch before Preview/H3 consumers, never substitute a
fake blank background. For NEW_VIEW/NEW_ENVIRONMENT or other pending resolution,
claim a wash, branch on `claimed`, pass its **nonempty token** through the external
washer, and attach with that token. Pass a real `removal_mask` whenever the washer
provides it. Automatic AttachClean cannot use the core's tokenless manual-override
API. Its READY image, or Resolve's reused READY image, can feed later consumers.

Wire Cleanup's mandatory `completion` input to a dependency after **all** final
consumers/workers (normally the final merged output chain). Cleanup is an output
node, not an unconnected utility, and ends only that video. Another video starts
empty even with identical pixels. Core handles are not shared across host worker
threads: independent workers must open their own core connection for the same
task and obey claim/token ownership. If an execution is cancelled or fails before
the cleanup sink, an owner must call `task.cleanup()` in its `finally` path; there
is no claimed host cancellation hook or hard-kill cleanup guarantee here.

Clean backgrounds are future additional H3 references, not RGB compositing or
per-Window washing. Segmentation and washer models are **not connected**; formal
JSON is **not wired**. RunningHub execution and H3 neural generation are **not
verified** by CPU tests.

Run `python -m unittest discover -s tests -p 'test_*.py'` with real Torch, NumPy
and qualified OpenCV for wrapper coverage. Background test imports fail visibly
when dependencies are missing, rather than reporting a skipped green suite.
The existing C builder regression writes an A2 fixture: set `H3_C_PROJECT_ROOT`
to a disposable copy containing `outputs/` and `work/modular_c_baseline_20261001/`
before running the full suite. Never point that test at production JSON directories.
