"""MiniMax H3 hard-cut scheduler for the formal motion-transfer workflow.

This node intentionally keeps the existing ComfyUI H3 worker nodes as the
execution primitives.  It only owns the dynamic part that a static graph
cannot express: hard-cut detection, synchronized RGB/Depth ranges, legal H3
length/padding, per-shot execution, deterministic trimming, and hard concat.

The node is deliberately dependency-light: PyTorch and the already installed
ComfyUI node registry are the only runtime requirements.  It does not import
Director/Easy, so the formal graph can keep using the current H3 core.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from typing import Any

import torch

try:  # ComfyUI's legacy node API
    import nodes
except Exception:  # pragma: no cover - import happens inside ComfyUI
    nodes = None


def _unwrap(value: Any) -> tuple:
    """Normalize legacy tuples and v3 NodeOutput objects to a tuple."""
    if hasattr(value, "args"):
        return tuple(value.args)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _call_node(class_type: str, values: dict[str, Any]) -> tuple:
    """Call a registered node while tolerating legacy/v3 signature drift."""
    if nodes is None or class_type not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError(f"Required ComfyUI node is not installed: {class_type}")
    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
    instance = cls()
    function_name = getattr(cls, "FUNCTION", "execute")
    fn = getattr(instance, function_name)
    signature = inspect.signature(fn)
    # V3 nodes expose FUNCTION as EXECUTE_NORMALIZED(_ASYNC), whose public
    # wrapper intentionally accepts **kwargs and then forwards to the real
    # classmethod execute().  Inspect the real method for filtering so legacy
    # flat/autogrow aliases cannot leak into the v3 call.
    if function_name in {"EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC"}:
        execute_fn = getattr(cls, "execute", None)
        if execute_fn is not None:
            signature = inspect.signature(execute_fn)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_kwargs:
        call_values = dict(values)
    else:
        call_values = {
            key: value
            for key, value in values.items()
            if key in signature.parameters
        }
    return _unwrap(fn(**call_values))


def _h3_ref2v(
    *,
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    picture1: torch.Tensor | None,
    picture2: torch.Tensor | None,
    depth: torch.Tensor | None,
):
    """Invoke the official H3 ReferenceToVideo node in either API shape."""
    refs = {}
    if picture1 is not None and int(picture1.shape[0]) > 0:
        refs["ref_image_0"] = picture1[:1]
    if picture2 is not None and int(picture2.shape[0]) > 0:
        refs["ref_image_1"] = picture2[:1]
    videos = {}
    if depth is not None and int(depth.shape[0]) > 0:
        videos["ref_video_0"] = depth
    out = _call_node(
        "MiniMaxH3ReferenceToVideo",
        {
            "clip": clip,
            "vae": vae,
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "ref_image_size": "match",
            "ref_images": refs,
            "ref_videos": videos,
            # Legacy versions expose flat autogrow names instead.
            **refs,
            **videos,
        },
    )
    if len(out) < 2:
        raise RuntimeError("MiniMaxH3ReferenceToVideo did not return conditioning + latent")
    return out[0], out[1]


def _derive_shot_seed(seed: int, shot_id: int) -> int:
    """Derive a legal ComfyUI/PyTorch uint64 seed for one shot."""
    base = int(seed)
    offset = int(shot_id)
    if base < 0 or base > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"seed must be in [0, 2**64-1], got {base}")
    if offset < 0:
        raise ValueError(f"shot_id must be non-negative, got {offset}")
    return (base + offset) & 0xFFFFFFFFFFFFFFFF

def _seeded_noise(seed: int):
    out = _call_node("RandomNoise", {"noise_seed": int(seed)})
    if not out:
        raise RuntimeError("RandomNoise returned no output")
    return out[0]


def _sample(model, positive, sampler, sigmas, latent, seed: int):
    guider = _call_node(
        "BasicGuider",
        {"model": model, "conditioning": positive},
    )[0]
    out = _call_node(
        "SamplerCustomAdvanced",
        {
            "noise": _seeded_noise(seed),
            "guider": guider,
            "sampler": sampler,
            "sigmas": sigmas,
            "latent_image": latent,
        },
    )
    if len(out) < 2:
        raise RuntimeError("SamplerCustomAdvanced returned fewer than two outputs")
    return out


def _upscale_video_latent(
    latent,
    *,
    model_name: str,
    target_width: int,
    target_height: int,
    align: int,
    device: str,
    precision: str,
    enable_chunking: bool,
):
    """Call the installed 3D upscaler through an explicit API adapter.

    The upstream node has shipped two incompatible callable contracts.  The
    current xmarre Plus source uses a dynamic-combo config object and
    ``keep_proportion``/``offload_after_upscale``.  The older LBH source uses
    the same config object but exposes
    ``enable_temporal_chunking``/``force_unload`` instead.  The RH-era LBH
    contract uses that same config object with a single ``enable_chunking``
    flag.  A legacy flat variant used a string mode plus plain width/height
    and ``enable_chunking``.  We select exactly one verified contract from the
    installed callable's signature; unknown signatures fail closed so a
    package update cannot silently change geometry or temporal behavior.
    """
    if nodes is None or "MinimaxH3LatentUpscaler3D" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("Required ComfyUI node is not installed: MinimaxH3LatentUpscaler3D")

    cls = nodes.NODE_CLASS_MAPPINGS["MinimaxH3LatentUpscaler3D"]
    instance = cls()
    function_name = getattr(cls, "FUNCTION", "execute")
    fn = getattr(instance, function_name)
    try:
        signature = inspect.signature(fn)
        # The real v3 class is called through EXECUTE_NORMALIZED, but API
        # detection must use the underlying classmethod execute signature.
        if function_name in {"EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC"}:
            execute_fn = getattr(cls, "execute", None)
            if execute_fn is not None:
                signature = inspect.signature(execute_fn)
        params = signature.parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect MinimaxH3LatentUpscaler3D "
            f"FUNCTION={function_name!r}: {exc}"
        ) from exc

    names = set(params)
    common = {"latent", "model_name", "mode", "align", "device", "precision"}
    mode = {
        "mode": "target dimensions",
        "width": int(target_width),
        "height": int(target_height),
    }

    if common.issubset(names) and "keep_proportion" in names:
        # xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus current API.
        call_values = {
            "latent": latent,
            "model_name": model_name,
            "mode": mode,
            "align": int(align),
            "keep_proportion": True,
            "device": device,
            "precision": precision,
        }
        if "offload_after_upscale" in names:
            call_values["offload_after_upscale"] = False
        api_name = "config/keep_proportion"
    elif common.issubset(names) and {
        "enable_temporal_chunking", "force_unload"
    }.issubset(names):
        # LBH upstream API with explicit temporal chunking and unload switches.
        call_values = {
            "latent": latent,
            "model_name": model_name,
            "mode": mode,
            "align": int(align),
            "enable_temporal_chunking": bool(enable_chunking),
            "force_unload": False,
            "device": device,
            "precision": precision,
        }
        api_name = "config/temporal_chunking"
    elif common.issubset(names) and "enable_chunking" in names and not {
        "width", "height"
    }.intersection(names):
        # RH-era LBH API (commit 6a4b191): dynamic-combo config plus one
        # enable_chunking switch.  Keep the workflow's explicit off setting.
        call_values = {
            "latent": latent,
            "model_name": model_name,
            "mode": mode,
            "align": int(align),
            "enable_chunking": bool(enable_chunking),
            "device": device,
            "precision": precision,
        }
        api_name = "config/chunking"
    elif common.issubset(names) and {"width", "height", "enable_chunking"}.issubset(names):
        # Legacy flat API.  Do not send dynamic-combo keys to this callable.
        call_values = {
            "latent": latent,
            "model_name": model_name,
            "mode": "target dimensions",
            "width": int(target_width),
            "height": int(target_height),
            "align": int(align),
            "device": device,
            "precision": precision,
            "enable_chunking": bool(enable_chunking),
        }
        if "offload_after_upscale" in names:
            call_values["offload_after_upscale"] = False
        api_name = "flat/legacy"
    else:
        raise RuntimeError(
            "Unsupported MinimaxH3LatentUpscaler3D API; "
            f"FUNCTION={function_name!r}, params={sorted(names)!r}"
        )

    try:
        out = _unwrap(fn(**call_values))
    except TypeError as exc:
        raise RuntimeError(
            "MinimaxH3LatentUpscaler3D call failed after selecting "
            f"API={api_name}: FUNCTION={function_name!r}, "
            f"params={sorted(names)!r}"
        ) from exc
    if not out:
        raise RuntimeError(
            "MinimaxH3LatentUpscaler3D returned no output: "
            f"FUNCTION={function_name!r}, API={api_name}"
        )
    return out[0]


def _fit_length(frames: torch.Tensor, target: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected IMAGE frames [T,H,W,C], got {tuple(frames.shape)}")
    if frames.shape[0] >= target:
        return frames[:target]
    if frames.shape[0] <= 0:
        raise ValueError("Cannot pad an empty shot")
    return torch.cat(
        [frames, frames[-1:].expand(target - frames.shape[0], -1, -1, -1)],
        dim=0,
    )


def _legal_length(frame_count: int) -> int:
    """First-round project rule: ceil to 17k+5, with a 39f floor."""
    f = max(1, int(frame_count))
    return max(39, int(math.ceil(max(0, f - 5) / 17.0) * 17 + 5))


def _detect_hard_cuts(rgb: torch.Tensor, threshold: float, min_shot_frames: int) -> list[tuple[int, int]]:
    """Deterministic cut detector using frame-to-frame RGB deltas.

    Runs of threshold crossings collapse to their strongest local peak.  A
    minimum shot length prevents motion spikes from creating one-frame shots.
    """
    total = int(rgb.shape[0])
    if total <= 1:
        return [(0, max(1, total))]
    x = rgb[..., :3]
    if x.dtype != torch.float32:
        x = x.float()
    chunk_size = 16
    delta_parts = []
    for start in range(0, total - 1, chunk_size):
        stop = min(total, start + chunk_size + 1)
        delta_parts.append(
            (x[start + 1 : stop] - x[start : stop - 1])
            .abs()
            .mean(dim=(1, 2, 3))
            .detach()
            .cpu()
        )
    delta = torch.cat(delta_parts, dim=0)
    high = delta >= float(threshold)
    candidates: list[int] = []
    i = 0
    while i < int(high.shape[0]):
        if not bool(high[i]):
            i += 1
            continue
        j = i + 1
        while j < int(high.shape[0]) and bool(high[j]):
            j += 1
        peak = i + int(torch.argmax(delta[i:j]).item())
        candidates.append(peak + 1)  # cut starts at the frame after the peak
        i = j
    cuts: list[int] = []
    last = 0
    min_len = max(1, int(min_shot_frames))
    for cut in candidates:
        if cut - last >= min_len and total - cut >= min_len:
            cuts.append(cut)
            last = cut
    edges = [0, *cuts, total]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1) if edges[k + 1] > edges[k]]



def _parse_scene_prompts(value: str) -> dict[int, dict[str, str]]:
    """Parse plain shot lines or JSONL records without losing shot metadata.

    Plain lines are assigned consecutive 1-based shot IDs.  JSONL records must
    carry an explicit positive ``shot`` and a non-empty string ``prompt``;
    malformed/ambiguous records fail closed instead of becoming prompt text.
    """
    lines = [raw.strip() for raw in str(value or "").splitlines() if raw.strip()]
    if not lines:
        raise ValueError("shot_prompts must contain one scene prompt per shot")
    structured = any(line.startswith("{") for line in lines)
    rows: dict[int, dict[str, str]] = {}
    for line_no, line in enumerate(lines, 1):
        if line.startswith("{"):
            if not structured:
                raise ValueError(f"shot_prompts line {line_no}: mixed formats are not allowed")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"shot_prompts line {line_no}: malformed JSON: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"shot_prompts line {line_no}: expected a JSON object")
            shot = obj.get("shot")
            if isinstance(shot, bool) or not isinstance(shot, int) or shot <= 0:
                raise ValueError(f"shot_prompts line {line_no}: shot must be a positive integer")
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"shot_prompts line {line_no}: prompt must be a non-empty string")
            if "subject_mode" not in obj or not isinstance(obj["subject_mode"], str):
                raise ValueError(f"shot_prompts line {line_no}: subject_mode is required in JSONL")
            subject_mode = obj["subject_mode"].strip().lower()
            if not subject_mode:
                raise ValueError(f"shot_prompts line {line_no}: subject_mode must be present or absent")
            if subject_mode not in {"present", "absent"}:
                raise ValueError(f"shot_prompts line {line_no}: subject_mode must be present or absent")
            record = {"prompt": prompt.strip(), "subject_mode": subject_mode}
        else:
            if structured:
                raise ValueError(f"shot_prompts line {line_no}: JSONL cannot be mixed with plain lines")
            shot = len(rows) + 1
            record = {
                "prompt": line,
                "subject_mode": "absent" if "subject_mode=absent" in line.lower() else "present",
            }
        if shot in rows:
            raise ValueError(f"shot_prompts contains duplicate shot id {shot}")
        rows[shot] = record
    return rows

def _validate_scene_prompts(scene_prompts: dict[int, dict[str, str]], shot_count: int) -> None:
    expected = set(range(1, int(shot_count) + 1))
    actual = set(scene_prompts)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "shot_prompts shot IDs must exactly match detected shots; "
            f"missing={missing}, extra={extra}"
        )

def _parse_empty_shots(value: str, shot_count: int | None = None) -> set[int]:
    """Parse the optional manual empty-shot list strictly.

    The list is retained only as a compatibility input.  Invalid tokens and
    indices outside the detected shot range fail closed instead of being
    silently ignored.
    """
    raw = str(value or "").strip()
    if not raw:
        return set()
    result = set()
    # Accept normal human forms such as ``1, 2``, ``1;2`` and ``1 2`` while
    # retaining fail-closed behaviour for malformed tokens.  Splitting on a
    # run of separators avoids manufacturing empty tokens from mixed
    # punctuation/whitespace.
    for token in re.split(r"[,;\s]+", raw):
        token = token.strip()
        if not token or not token.isascii() or not token.isdigit():
            raise ValueError(f"empty_shot_indices contains invalid token: {token!r}")
        index = int(token)
        if index < 1:
            raise ValueError(f"empty_shot_indices index must be >= 1: {index}")
        if shot_count is not None and index > int(shot_count):
            raise ValueError(
                f"empty_shot_indices index {index} exceeds detected shot count {int(shot_count)}"
            )
        result.add(index)
    return result

def _resolve_empty_shot(scene_record, shot_number, manual_empty_shots):
    """Resolve metadata/manual empty state without silent conflicting OR."""
    mode = str(scene_record["subject_mode"]).strip().lower()
    manual = int(shot_number) in manual_empty_shots
    if manual and mode == "present":
        raise ValueError(
            f"empty_shot_indices conflicts with subject_mode=present for shot {int(shot_number)}"
        )
    return mode == "absent" or manual

def _compose_shot_prompt(scene_prompts, shot_id, global_prompt, is_empty, pass_name):
    scene = scene_prompts[shot_id + 1]["prompt"]
    if is_empty:
        return (
            f"{scene}\nEmpty-shot policy: no person, no human silhouette, no mannequin, "
            f"no face, no clothing, no identity reference. This is the {pass_name} pass; "
            "preserve only environment, camera, depth and atmosphere."
        )
    return f"{scene}\nShot-local motion/reference constraints ({pass_name} pass): {global_prompt}"


class H3HardCutRunnerShotPromptEmptyShot:
    """Dynamic hard-cut wrapper around the current formal H3 worker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "rgb": ("IMAGE",),
                "depth": ("IMAGE",),
                "picture1": ("IMAGE",),
                "picture2": ("IMAGE",),
                "prompt_first": ("STRING", {"multiline": True, "default": ""}),
                "prompt_second": ("STRING", {"multiline": True, "default": ""}),
                "sampler": ("SAMPLER",),
                "high_sigmas": ("SIGMAS",),
                "low_sigmas": ("SIGMAS",),
                "base_width": ("INT", {"default": 672, "min": 32, "step": 32}),
                "base_height": ("INT", {"default": 1184, "min": 32, "step": 32}),
                "target_width": ("INT", {"default": 1088, "min": 32, "step": 32}),
                "target_height": ("INT", {"default": 1920, "min": 32, "step": 32}),
                "upscaler_model": ("STRING", {"default": "minimax_h3_latent_upscaler_3d_fp16.safetensors"}),
                "seed": ("INT", {"default": 999, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cut_threshold": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_shot_frames": ("INT", {"default": 8, "min": 1, "max": 240}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "upscaler_align": ("INT", {"default": 32, "min": 16, "step": 16}),
                "upscaler_device": (["cuda", "cpu"], {"default": "cuda"}),
                "upscaler_precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "enable_upscaler_chunking": ("BOOLEAN", {"default": False}),
                "shot_prompts": ("STRING", {"multiline": True, "default": ""}),
                "empty_shot_indices": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "fps", "frame_count", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3"

    def run(
        self,
        model,
        clip,
        video_vae,
        rgb,
        depth,
        picture1,
        picture2,
        prompt_first,
        prompt_second,
        sampler,
        high_sigmas,
        low_sigmas,
        base_width,
        base_height,
        target_width,
        target_height,
        upscaler_model,
        seed,
        cut_threshold,
        min_shot_frames,
        fps,
        upscaler_align,
        upscaler_device,
        upscaler_precision,
        enable_upscaler_chunking,
        shot_prompts,
        empty_shot_indices,
    ):
        if int(rgb.shape[0]) != int(depth.shape[0]):
            raise ValueError(
                f"RGB/Depth frame count mismatch: {int(rgb.shape[0])} vs {int(depth.shape[0])}"
            )
        shots = _detect_hard_cuts(rgb, cut_threshold, min_shot_frames)
        scene_prompts = _parse_scene_prompts(shot_prompts)
        empty_shots = _parse_empty_shots(empty_shot_indices, shot_count=len(shots))
        _validate_scene_prompts(scene_prompts, len(shots))
        total_frames = sum(int(end - start) for start, end in shots)
        output_images: torch.Tensor | None = None
        output_cursor = 0
        report_lines = [
            "H3HardCutRunner: hard-cut isolated execution",
            f"shots={len(shots)}, fps={float(fps):.6g}, threshold={float(cut_threshold):.4g}",
        ]
        for shot_id, (start, end) in enumerate(shots):
            original_f = int(end - start)
            scene_record = scene_prompts[shot_id + 1]
            is_empty = _resolve_empty_shot(scene_record, shot_id + 1, empty_shots)
            shot_seed = _derive_shot_seed(seed, shot_id)
            work_l = _legal_length(original_f)
            depth_shot = _fit_length(depth[start:end], work_l)
            first_positive, first_latent = _h3_ref2v(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(scene_prompts, shot_id, str(prompt_first or ""), is_empty, "first"),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=None if is_empty else picture1,
                picture2=None if is_empty else picture2,
                depth=depth_shot,
            )
            _first_output, first_clean = _sample(
                model, first_positive, sampler, high_sigmas, first_latent, shot_seed
            )
            split = _call_node("LTXVSeparateAVLatent", {"av_latent": first_clean})
            if len(split) < 2:
                raise RuntimeError("LTXVSeparateAVLatent returned fewer than two streams")
            upscaled_video = _upscale_video_latent(
                split[0],
                model_name=str(upscaler_model),
                target_width=int(target_width),
                target_height=int(target_height),
                align=int(upscaler_align),
                device=str(upscaler_device),
                precision=str(upscaler_precision),
                enable_chunking=bool(enable_upscaler_chunking),
            )
            joined = _call_node(
                "LTXVConcatAVLatent",
                {"video_latent": upscaled_video, "audio_latent": split[1]},
            )[0]
            second_positive, _unused_latent = _h3_ref2v(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(scene_prompts, shot_id, str(prompt_second or ""), is_empty, "second"),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=None if is_empty else picture1,
                picture2=None if is_empty else picture2,
                depth=None,
            )
            second_output, _second_clean = _sample(
                model, second_positive, sampler, low_sigmas, joined, shot_seed
            )
            decoded = _call_node("VAEDecode", {"samples": second_output, "vae": video_vae})[0]
            trimmed = decoded[:original_f].detach().to(device="cpu", dtype=torch.float32)
            if output_images is None:
                output_images = torch.empty(
                    (total_frames, *tuple(trimmed.shape[1:])),
                    dtype=trimmed.dtype,
                    device="cpu",
                )
            elif tuple(output_images.shape[1:]) != tuple(trimmed.shape[1:]):
                raise RuntimeError("Shot decode shape changed across hard-cut segments")
            output_images[output_cursor : output_cursor + original_f].copy_(trimmed)
            output_cursor += original_f
            report_lines.append(
                f"shot={shot_id + 1} range=[{start},{end}) F={original_f} L={work_l} "
                f"subject={'absent' if is_empty else 'present'} prompt=shot-local "
                "continuity=OFF trim=deterministic"
            )
            del (
                depth_shot,
                first_positive,
                first_latent,
                _first_output,
                first_clean,
                split,
                upscaled_video,
                joined,
                second_positive,
                _unused_latent,
                second_output,
                _second_clean,
                decoded,
                trimmed,
            )
        if output_images is None:
            raise RuntimeError("H3HardCutRunner produced no decoded shot")
        return (output_images, float(fps), int(output_images.shape[0]), "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {"H3HardCutRunnerShotPromptEmptyShot": H3HardCutRunnerShotPromptEmptyShot}
NODE_DISPLAY_NAME_MAPPINGS = {"H3HardCutRunnerShotPromptEmptyShot": "H3 Hard-Cut Runner A (Shot Prompt + Empty Shot)"}
