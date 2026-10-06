"""Small adapters for connecting H3 Modular-C shot scheduling to REF2VA/SelfLift graphs."""

from __future__ import annotations

import json
import math
from typing import Any

import torch

UINT64_MAX = 0xFFFFFFFFFFFFFFFF
MAX_RELATIVE_ASPECT_DRIFT = 0.01


def _require_uint64_seed(value: Any, name: str = "shot seed") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer in uint64 range")
    try:
        seed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer in uint64 range") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer in uint64 range")
    if seed < 0 or seed > UINT64_MAX:
        raise ValueError(f"{name} must be in uint64 range 0..{UINT64_MAX}")
    return seed


def _optional_module_names(module_name: str) -> set[str]:
    names = {module_name}
    if __package__:
        names.add(f"{__package__}.{module_name}")
    return names


# The complete plugin supplies these private helpers.  The explicit fallback is
# only for this review candidate, which intentionally does not bundle the old
# runner files; it preserves the same manifest/tail-hold seam for contract tests.
try:
    from .h3_hardcut_modular_c import _manifest  # type: ignore
except ModuleNotFoundError as exc:
    if exc.name not in _optional_module_names("h3_hardcut_modular_c"):
        raise

    def _manifest(value: str) -> dict:
        try:
            obj = json.loads(str(value))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest must be valid JSON") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("shots"), list):
            raise ValueError("manifest requires a shots list")
        shots = obj["shots"]
        try:
            total = int(obj["total_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("manifest requires total_frames") from exc
        if total <= 0 or not shots:
            raise ValueError("manifest requires non-empty shots and positive total_frames")
        cursor = 0
        for expected, shot in enumerate(shots, 1):
            if not isinstance(shot, dict):
                raise ValueError("manifest shots must be objects")
            try:
                shot_id = int(shot["shot"])
                start = int(shot["start"])
                end = int(shot["end"])
                original_f = int(shot["original_f"])
                work_l = int(shot["work_l"])
                _require_uint64_seed(shot["seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("manifest shot fields are invalid and seed is required") from exc
            if shot_id != expected or start != cursor or end <= start or end > total:
                raise ValueError("manifest shots must have contiguous IDs and ranges")
            if original_f != end - start or original_f <= 0 or work_l < original_f:
                raise ValueError("manifest shot length fields are inconsistent")
            if work_l != _legal_length(original_f):
                raise ValueError("manifest work_l must equal production _legal_length(original_f)")
            cursor = end
        if cursor != total:
            raise ValueError("manifest shot ranges must cover total_frames")
        return obj

try:
    from .h3_hardcut_runner_a import _fit_length, _legal_length  # type: ignore
except ModuleNotFoundError as exc:
    if exc.name not in _optional_module_names("h3_hardcut_runner_a"):
        raise

    def _legal_length(frame_count: int) -> int:
        """Match the production H3 first-round 17k+5 rule, including its 39f floor."""
        f = max(1, int(frame_count))
        return max(39, int(math.ceil(max(0, f - 5) / 17.0) * 17 + 5))

    def _fit_length(value: torch.Tensor, length: int) -> torch.Tensor:
        current = int(value.shape[0])
        target = int(length)
        if target < 1 or current < 1:
            raise ValueError("cannot fit an empty video")
        if current == target:
            return value
        if current > target:
            return value[:target]
        tail = value[-1:]
        return torch.cat((value, tail.expand(target - current, *tail.shape[1:])), dim=0)


CATEGORY = "video/MiniMaxH3/C Modular/Adapters"
MAX_H3_LENGTH = 3600
MIN_GENERATION_WINDOW = 124
EXPECTED_FPS = 24


def _require_image_batch(value: torch.Tensor, name: str) -> None:
    if value is None or not hasattr(value, "ndim") or int(value.ndim) != 4:
        shape = None if value is None else getattr(value, "shape", None)
        raise ValueError(f"{name} must be IMAGE [T,H,W,C], got {shape}")
    if int(value.shape[0]) <= 0:
        raise ValueError(f"{name} is empty")
    if int(value.shape[1]) <= 0 or int(value.shape[2]) <= 0 or int(value.shape[3]) < 3:
        raise ValueError(f"{name} must be a ComfyUI IMAGE with positive H/W and at least 3 channels, got {tuple(value.shape)}")


def _require_matching_frame_count(first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str) -> None:
    if int(first.shape[0]) != int(second.shape[0]):
        raise ValueError(
            f"{first_name}/{second_name} frame-count mismatch: "
            f"{int(first.shape[0])} != {int(second.shape[0])}"
        )


def _official_canvas_bucket(width: int, height: int) -> tuple[int, int]:
    """Mirror official H3 adapt_canvas rounding for geometry compatibility."""
    canvas_multiple = 32
    base_short_edge = 768
    max_pixels = 768 * 1344
    ratio = float(width) / float(height)
    if ratio >= 1.0:
        nominal_width, nominal_height = base_short_edge * ratio, base_short_edge
    else:
        nominal_width, nominal_height = base_short_edge, base_short_edge / ratio
    if nominal_width * nominal_height > max_pixels:
        scale = math.sqrt(max_pixels / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return (
        max(canvas_multiple, round(nominal_width / canvas_multiple) * canvas_multiple),
        max(canvas_multiple, round(nominal_height / canvas_multiple) * canvas_multiple),
    )


def _require_matching_aspect_ratio(first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str) -> None:
    first_height, first_width = int(first.shape[1]), int(first.shape[2])
    second_height, second_width = int(second.shape[1]), int(second.shape[2])
    first_canvas = _official_canvas_bucket(first_width, first_height)
    second_canvas = _official_canvas_bucket(second_width, second_height)
    first_ratio = float(first_width) / float(first_height)
    second_ratio = float(second_width) / float(second_height)
    relative_drift = abs(first_ratio - second_ratio) / max(first_ratio, second_ratio)
    if first_canvas != second_canvas or relative_drift > MAX_RELATIVE_ASPECT_DRIFT:
        raise ValueError(
            f"{first_name}/{second_name} aspect-ratio mismatch: "
            f"{first_height}x{first_width} ratio={first_ratio:.8f} canvas={first_canvas}; "
            f"{second_height}x{second_width} ratio={second_ratio:.8f} canvas={second_canvas}; "
            f"relative_drift={relative_drift:.6f}"
        )


def _require_matching_layout(first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str) -> None:
    if tuple(first.shape[1:]) != tuple(second.shape[1:]):
        raise ValueError(
            f"{first_name}/{second_name} H/W/C mismatch: "
            f"{tuple(first.shape[1:])} != {tuple(second.shape[1:])}"
        )
    first_dtype = getattr(first, "dtype", None)
    second_dtype = getattr(second, "dtype", None)
    if first_dtype is not None and second_dtype is not None and first_dtype != second_dtype:
        raise ValueError(f"{first_name}/{second_name} dtype mismatch: {first_dtype} != {second_dtype}")
    first_device = getattr(first, "device", None)
    second_device = getattr(second, "device", None)
    if first_device is not None and second_device is not None and first_device != second_device:
        raise ValueError(f"{first_name}/{second_name} device mismatch: {first_device} != {second_device}")


def _require_aggregate_length(length: int, name: str) -> None:
    if length < 39 or length % 17 != 5:
        raise ValueError(f"{name} must be on the MiniMax H3 17k+5 frame grid, got {length}")


def _require_generation_length(length: int, name: str) -> None:
    _require_aggregate_length(length, name)
    if length < MIN_GENERATION_WINDOW:
        raise ValueError(f"{name} must be at least {MIN_GENERATION_WINDOW} for the donor H3 generation contract")
    if length > MAX_H3_LENGTH:
        raise ValueError(f"{name} must be <= {MAX_H3_LENGTH} for one MiniMax H3 generation, got {length}")


def _row_int(row: dict, key: str) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"window manifest field {key!r} is invalid") from exc


def _parse_window_manifest(value: str) -> dict:
    try:
        obj = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("window manifest must be valid JSON") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("windows"), list):
        raise ValueError("window manifest requires a windows list")
    try:
        work_l = int(obj.get("shot_work_l", 0))
        window_size = int(obj.get("window_size", 0))
        overlap = int(obj.get("overlap", -1))
        stride = int(obj.get("stride", window_size - overlap))
    except (TypeError, ValueError) as exc:
        raise ValueError("window manifest header is invalid") from exc
    _require_aggregate_length(work_l, "shot_work_l")
    _require_generation_length(window_size, "window_size")
    if window_size < MIN_GENERATION_WINDOW:
        raise ValueError(f"window_size must be at least {MIN_GENERATION_WINDOW} for the donor H3 generation contract")
    if overlap < 0 or overlap >= window_size or stride != window_size - overlap:
        raise ValueError("window manifest header stride/overlap is invalid")
    if overlap not in {0, 1} and overlap % 17 != 5:
        raise ValueError("window manifest overlap must be 17k+5 or 0/1")
    windows = obj["windows"]
    if not windows:
        raise ValueError("window manifest has no windows")

    append_cursor = 0
    previous_end = None
    for expected, row in enumerate(windows, 1):
        if not isinstance(row, dict) or _row_int(row, "window") != expected:
            raise ValueError("window manifest IDs must be ordered and contiguous")
        start = _row_int(row, "start")
        source_count = _row_int(row, "source_count")
        condition_length = _row_int(row, "condition_length")
        append_start = _row_int(row, "append_start")
        append_count = _row_int(row, "append_count")
        guide_frames = _row_int(row, "guide_frames")
        pad_tail = _row_int(row, "pad_tail")
        if start < 0 or source_count <= 0 or condition_length < source_count:
            raise ValueError("window manifest range is invalid")
        if start + source_count > work_l:
            raise ValueError("window manifest reads past shot_work_l")
        if append_start < 0 or append_count <= 0 or append_start + append_count > source_count:
            raise ValueError("window manifest append range is invalid")
        if guide_frames < 0 or guide_frames > source_count:
            raise ValueError("window manifest guide range is invalid")
        if pad_tail != condition_length - source_count:
            raise ValueError("window manifest pad_tail is inconsistent")
        if condition_length != window_size:
            raise ValueError("window manifest condition_length is inconsistent")
        if expected == 1:
            if start != 0 or guide_frames != 0 or append_start != 0:
                raise ValueError("first window must start at zero without a guide overlap")
        else:
            if start != (expected - 2) * stride + stride:
                raise ValueError("window starts must advance by the declared stride")
            if previous_end is None or previous_end - start != guide_frames:
                raise ValueError("window guide_frames do not match the actual previous-window overlap")
            if guide_frames != overlap or append_start != guide_frames:
                raise ValueError("window guide/append ranges do not match overlap")
        if append_start != guide_frames or append_count != source_count - guide_frames:
            raise ValueError("window append range must follow the guide range")
        global_start = start + append_start
        if global_start != append_cursor:
            raise ValueError("window append ranges must be contiguous and gap-free")
        append_cursor += append_count
        previous_end = start + source_count

    if append_cursor != work_l:
        raise ValueError("window append ranges must cover exactly shot_work_l frames")
    return obj


class H3CShotDualRefPad:
    """Select one Modular-C shot and expose synchronized raw/padded RGB+Depth."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"rgb": ("IMAGE",), "depth": ("IMAGE",), "manifest": ("STRING", {"forceInput": True}), "index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("shot_rgb_raw", "rgb_ref_video", "depth_ref_video", "original_f", "work_l", "shot_seed", "shot_number", "range", "pad_tail_frames", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, rgb, depth, manifest, index):
        _require_image_batch(rgb, "rgb")
        _require_image_batch(depth, "depth")
        _require_matching_frame_count(rgb, depth, "rgb", "depth")
        _require_matching_aspect_ratio(rgb, depth, "rgb", "depth")
        obj = _manifest(manifest)
        try:
            total = int(obj["total_frames"])
            shots = obj["shots"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("manifest requires total_frames and shots") from exc
        if int(rgb.shape[0]) != total or int(depth.shape[0]) != total:
            raise ValueError(f"RGB/Depth frame counts no longer match the Modular-C shot manifest: rgb={int(rgb.shape[0])}, depth={int(depth.shape[0])}, manifest={total}")
        i = int(index)
        if i < 0 or i >= len(shots):
            raise ValueError(f"shot index out of range: {i}")
        shot = shots[i]
        try:
            shot_seed = _require_uint64_seed(shot["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("shot manifest seed is required and must be uint64") from exc
        try:
            start = int(shot["start"]); end = int(shot["end"]); original_f = int(shot["original_f"]); work_l = int(shot["work_l"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("shot manifest range/work length fields are invalid") from exc
        _require_aggregate_length(work_l, "shot work_l")
        if start < 0 or end > total or original_f != end - start or original_f <= 0 or work_l < original_f:
            raise ValueError("shot manifest range/work length contract changed")
        shot_rgb_raw = rgb[start:end]
        shot_depth_raw = depth[start:end]
        if int(shot_rgb_raw.shape[0]) != original_f or int(shot_depth_raw.shape[0]) != original_f:
            raise ValueError("selected RGB/Depth shot length mismatch")
        rgb_ref_video = _fit_length(shot_rgb_raw, work_l)
        depth_ref_video = _fit_length(shot_depth_raw, work_l)
        if int(rgb_ref_video.shape[0]) != work_l or int(depth_ref_video.shape[0]) != work_l:
            raise RuntimeError("dual reference padding did not produce work_l frames")
        pad_tail = work_l - original_f
        number = i + 1
        report = (f"shot={number} range=[{start},{end}) original_f={original_f} work_l={work_l} pad_tail={pad_tail} "
                  f"pad_policy=hold_current_shot_last_frame expected_upstream_fps={EXPECTED_FPS} "
                  "rgb_role=Video1_motion_camera_scene depth_role=Video2_spatial_geometry")
        return (shot_rgb_raw, rgb_ref_video, depth_ref_video, original_f, work_l, shot_seed, number, f"[{start},{end})", pad_tail, report)


class H3CShotGuideWindowPlanner:
    """Plan donor-style AddGuide windows inside one shot only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"work_l": ("INT", {"forceInput": True, "min": 39}), "window_size": ("INT", {"default": 124, "min": 124, "step": 17}), "overlap": ("INT", {"default": 22, "min": 0, "step": 1})}}

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("window_manifest", "window_count", "stride", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, work_l, window_size, overlap):
        length, window, guide = int(work_l), int(window_size), int(overlap)
        _require_aggregate_length(length, "work_l")
        _require_generation_length(window, "window_size")
        if guide < 0 or guide >= window:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")
        if guide not in {0, 1} and guide % 17 != 5:
            raise ValueError("multi-frame overlap must be 17k+5 (e.g. 22, 39, 56...) or 0/1")
        stride = window - guide
        rows = []
        if length <= window:
            rows.append({"window": 1, "start": 0, "source_count": length, "condition_length": window, "append_start": 0, "append_count": length, "guide_frames": 0, "pad_tail": window - length})
        else:
            count = 1 + int(math.ceil((length - window) / float(stride)))
            for idx in range(count):
                start = idx * stride
                source_count = min(window, length - start)
                append_start = 0 if idx == 0 else guide
                append_count = source_count - append_start
                if append_count <= 0:
                    raise RuntimeError("AddGuide overlap consumed an entire window")
                rows.append({"window": idx + 1, "start": start, "source_count": source_count, "condition_length": window, "append_start": append_start, "append_count": append_count, "guide_frames": 0 if idx == 0 else guide, "pad_tail": window - source_count})
        obj = {"shot_work_l": length, "window_size": window, "overlap": guide, "stride": stride, "windows": rows}
        serialised = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        _parse_window_manifest(serialised)
        produced = sum(int(row["append_count"]) for row in rows)
        if produced != length:
            raise RuntimeError(f"window append contract mismatch: produced={produced}, work_l={length}")
        return serialised, len(rows), stride, f"shot_work_l={length} windows={len(rows)} window_size={window} overlap={guide} stride={stride} append_total={produced}; AddGuide resets at every hard cut"


class H3CShotGuideWindowSelect:
    """Select synchronized RGB/Depth inputs for one planned AddGuide window."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"rgb_ref_video": ("IMAGE",), "depth_ref_video": ("IMAGE",), "window_manifest": ("STRING", {"forceInput": True}), "index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "INT", "INT", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("window_rgb", "window_depth", "condition_length", "append_start", "append_count", "guide_frames", "append_offset", "window_number", "needs_addguide", "is_first", "is_last", "range", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, rgb_ref_video, depth_ref_video, window_manifest, index):
        _require_image_batch(rgb_ref_video, "rgb_ref_video")
        _require_image_batch(depth_ref_video, "depth_ref_video")
        _require_matching_frame_count(rgb_ref_video, depth_ref_video, "rgb_ref_video", "depth_ref_video")
        _require_matching_aspect_ratio(rgb_ref_video, depth_ref_video, "rgb_ref_video", "depth_ref_video")
        obj = _parse_window_manifest(window_manifest)
        work_l = int(obj["shot_work_l"])
        if int(rgb_ref_video.shape[0]) != work_l or int(depth_ref_video.shape[0]) != work_l:
            raise ValueError(f"window inputs must equal shot_work_l before window slicing: rgb={int(rgb_ref_video.shape[0])}, depth={int(depth_ref_video.shape[0])}, work_l={work_l}")
        i = int(index); rows = obj["windows"]
        if i < 0 or i >= len(rows):
            raise ValueError(f"window index out of range: {i}")
        row = rows[i]; start = int(row["start"]); source_count = int(row["source_count"]); end = start + source_count; condition_length = int(row["condition_length"])
        rgb = _fit_length(rgb_ref_video[start:end], condition_length)
        depth = _fit_length(depth_ref_video[start:end], condition_length)
        guide_frames = int(row["guide_frames"]); append_start = int(row["append_start"]); append_count = int(row["append_count"]); append_offset = start + append_start
        report = f"window={i + 1}/{len(rows)} source=[{start},{end}) condition_length={condition_length} guide_frames={guide_frames} append=[{append_start},{append_start + append_count}) append_offset={append_offset} pad_tail={int(row['pad_tail'])}"
        return (rgb, depth, condition_length, append_start, append_count, guide_frames, append_offset, i + 1, guide_frames > 0, i == 0, i == len(rows) - 1, f"[{start},{end})", report)


class H3CShotGuideWindowAppend:
    """Append only generated, non-guide frames and verify gap-free shot state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_window": ("IMAGE",),
                "condition_length": ("INT", {"forceInput": True, "min": 5}),
                "append_start": ("INT", {"forceInput": True, "min": 0}),
                "append_count": ("INT", {"forceInput": True, "min": 1}),
                "append_offset": ("INT", {"forceInput": True, "min": 0}),
                "expected_total": ("INT", {"forceInput": True, "min": 1}),
                "is_first": ("BOOLEAN", {"forceInput": True}),
                "is_last": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
                "accumulated": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("accumulated", "appended_count", "complete", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, generated_window, condition_length, append_start, append_count, append_offset, expected_total, is_first, is_last, accumulated=None):
        _require_image_batch(generated_window, "generated_window")
        expected_window = int(condition_length)
        _require_generation_length(expected_window, "condition_length")
        if int(generated_window.shape[0]) != expected_window:
            raise ValueError(
                f"generated_window length {int(generated_window.shape[0])} does not equal condition_length {expected_window}"
            )
        start, count, offset, total = int(append_start), int(append_count), int(append_offset), int(expected_total)
        _require_aggregate_length(total, "expected_total")
        if start < 0 or count <= 0 or start + count > int(generated_window.shape[0]):
            raise ValueError("append range is outside generated_window")
        if offset < 0 or total <= 0 or offset + count > total:
            raise ValueError("append offset exceeds expected shot length")
        if bool(is_first):
            if offset != 0 or accumulated is not None:
                raise ValueError("first append must start at zero without accumulated state")
            result = generated_window[start:start + count]
        else:
            if accumulated is None:
                raise ValueError("non-first append requires accumulated state")
            _require_image_batch(accumulated, "accumulated")
            if int(accumulated.shape[0]) != offset:
                raise ValueError(f"accumulated length {int(accumulated.shape[0])} does not equal append_offset {offset}")
            _require_matching_layout(accumulated, generated_window, "accumulated", "generated_window")
            result = torch.cat((accumulated, generated_window[start:start + count]), dim=0)
        complete = int(result.shape[0]) == total
        if bool(is_last) != complete:
            raise ValueError(f"is_last={bool(is_last)} disagrees with complete={complete}")
        return result, count, complete, f"append_offset={offset} append_count={count} accumulated={int(result.shape[0])}/{total} complete={complete}"


NODE_CLASS_MAPPINGS = {
    "H3CShotDualRefPad": H3CShotDualRefPad,
    "H3CShotGuideWindowPlanner": H3CShotGuideWindowPlanner,
    "H3CShotGuideWindowSelect": H3CShotGuideWindowSelect,
    "H3CShotGuideWindowAppend": H3CShotGuideWindowAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CShotDualRefPad": "H3 C Shot Dual Ref Pad (RGB + Depth)",
    "H3CShotGuideWindowPlanner": "H3 C Shot AddGuide Window Planner",
    "H3CShotGuideWindowSelect": "H3 C Shot AddGuide Window Select",
    "H3CShotGuideWindowAppend": "H3 C Shot AddGuide Window Append",
}
