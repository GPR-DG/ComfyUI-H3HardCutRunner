# ComfyUI-H3HardCutRunner

`H3HardCutRunner` is a ComfyUI custom node for deterministic MiniMax H3
hard-cut processing. It detects hard cuts, keeps RGB and depth ranges aligned,
rounds each shot to a legal H3 length, runs the configured two-pass H3 path,
performs the learned 3D latent upscale between passes, and trims/concatenates
the decoded shot results.

The node keeps seed and shot identity deterministic, uses the installed
ComfyUI node registry as its execution boundary, and adapts the installed
`MinimaxH3LatentUpscaler3D` callable by inspected signature. It supports the
verified current xmarre configuration-dictionary contract, the RH-era LBH
configuration-dictionary plus `enable_chunking` contract, the newer temporal-
chunking contract, and the legacy flat contract. It fails closed for an
unknown signature. Temporal chunking remains controlled by the runner setting,
while the project default is disabled.

Install by copying this directory into `ComfyUI/custom_nodes/` and restarting
ComfyUI. The workflow must provide the standard MiniMax H3 model, CLIP, video
VAE, sampler, sigma schedules, RGB/depth inputs, two reference images, and
the learned 3D upscaler checkpoint. The node returns decoded images, FPS,
frame count, and a diagnostic report; it does not expose an AUDIO output.

