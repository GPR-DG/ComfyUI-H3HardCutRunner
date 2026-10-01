"""Observable C-only shot contracts around the existing H3 worker nodes.

These nodes deliberately do not call H3 conditioning, sampling, upscaling, or
VAE decoding.  A canvas loop must connect those registered workers explicitly.
"""

from __future__ import annotations

import json

import torch

from .h3_hardcut_runner_a import (
    _derive_shot_seed,
    _detect_hard_cuts,
    _fit_length,
    _legal_length,
    _parse_empty_shots,
    _validate_scene_prompts,
)
from .h3_hardcut_runner_c import (
    _compose_shot_prompt,
    _has_reference_tensor,
    _parse_manual_force_empty_indices,
    _parse_scene_prompts,
    _resolve_shot_policy,
)


def _manifest(value: str) -> dict:
    try:
        obj = json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("shot manifest must be valid JSON") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("shots"), list):
        raise ValueError("shot manifest requires a shots list")
    shots = obj["shots"]
    if not shots:
        raise ValueError("shot manifest has no shots")
    cursor = 0
    for number, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            raise ValueError("shot manifest row must be an object")
        if int(shot["shot"]) != number or int(shot["start"]) != cursor:
            raise ValueError("shot manifest IDs/ranges must be ordered and contiguous")
        end = int(shot["end"])
        if end <= cursor or int(shot["original_f"]) != end - cursor:
            raise ValueError("shot manifest frame range is invalid")
        if int(shot["work_l"]) != _legal_length(end - cursor):
            raise ValueError("shot manifest work length changed")
        cursor = end
    if cursor != int(obj["total_frames"]):
        raise ValueError("shot manifest total frame count mismatch")
    return obj


class H3CShotPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgb": ("IMAGE",), "depth": ("IMAGE",),
            "seed": ("INT", {"default": 999, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "cut_threshold": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0}),
            "min_shot_frames": ("INT", {"default": 8, "min": 1, "max": 240}),
        }}

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("manifest", "shot_count", "total_frames", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, rgb, depth, seed, cut_threshold, min_shot_frames):
        total = int(rgb.shape[0])
        if total <= 0 or total != int(depth.shape[0]):
            raise ValueError("RGB/Depth frame count mismatch or empty source")
        ranges = _detect_hard_cuts(rgb, cut_threshold, min_shot_frames)
        shots = []
        for shot_id, (start, end) in enumerate(ranges):
            original_f = int(end - start)
            shots.append({
                "shot": shot_id + 1, "start": int(start), "end": int(end),
                "original_f": original_f, "work_l": _legal_length(original_f),
                "seed": _derive_shot_seed(seed, shot_id),
            })
        manifest = {"total_frames": total, "shots": shots}
        serialised = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
        _manifest(serialised)
        report = "\n".join(
            f"shot={s['shot']} range=[{s['start']},{s['end']}) F={s['original_f']} L={s['work_l']} seed={s['seed']}"
            for s in shots
        )
        return serialised, len(shots), total, report


class H3CShotSelectPad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"rgb": ("IMAGE",), "depth": ("IMAGE",),
                             "manifest": ("STRING", {"forceInput": True}),
                             "index": ("INT", {"default": 0, "min": 0})}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("shot_rgb", "depth_ref_video", "original_f", "work_l", "shot_seed", "shot_number", "range")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, rgb, depth, manifest, index):
        obj = _manifest(manifest)
        if int(rgb.shape[0]) != int(obj["total_frames"]) or int(depth.shape[0]) != int(obj["total_frames"]):
            raise ValueError("RGB/Depth no longer match the shot manifest")
        i = int(index)
        if i < 0 or i >= len(obj["shots"]):
            raise ValueError(f"shot index out of range: {i}")
        shot = obj["shots"][i]
        start, end = int(shot["start"]), int(shot["end"])
        work_l = int(shot["work_l"])
        depth_ref = _fit_length(depth[start:end], work_l)
        return (rgb[start:end], depth_ref, int(shot["original_f"]), work_l,
                int(shot["seed"]), i + 1, f"[{start},{end})")


class H3CShotPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_prompts": ("STRING", {"forceInput": True}),
            "empty_shot_indices": ("STRING", {"forceInput": True}),
            "shot_number": ("INT", {"forceInput": True}),
            "shot_count": ("INT", {"forceInput": True}),
            "seed": ("INT", {"forceInput": True}),
            "picture1": ("IMAGE",), "picture2": ("IMAGE",),
        }, "optional": {
            "picture3": ("IMAGE",),
            "manual_force_empty_indices": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BOOLEAN", "BOOLEAN", "BOOLEAN",
                    "STRING", "STRING", "STRING", "STRING", "STRING", "BOOLEAN",
                    "BOOLEAN", "BOOLEAN", "BOOLEAN", "INT")
    RETURN_NAMES = ("picture1", "picture2", "picture3", "strict_empty", "allow_secondary_humans",
                    "inject_target_references", "subject_mode", "secondary_human_presence",
                    "source_prop_policy", "reason", "report", "picture3_present",
                    "picture1_used", "picture2_used", "picture3_used", "seed")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, shot_prompts, empty_shot_indices, shot_number, shot_count, seed, picture1, picture2,
            picture3=None, manual_force_empty_indices=""):
        rows = _parse_scene_prompts(shot_prompts)
        _validate_scene_prompts(rows, int(shot_count))
        number = int(shot_number)
        if number not in rows:
            raise ValueError(f"shot {number} is missing from scene records")
        empty = _parse_empty_shots(empty_shot_indices, shot_count=int(shot_count))
        empty |= _parse_manual_force_empty_indices(manual_force_empty_indices, shot_count=int(shot_count))
        record = rows[number]
        policy = _resolve_shot_policy(record, number, empty)
        inject = bool(policy["inject_target_references"])
        refs = tuple(
            value if inject and _has_reference_tensor(value) else None
            for value in (picture1, picture2, picture3)
        )
        mode = str(record["subject_mode"])
        secondary = str(record.get("secondary_human_presence", "none"))
        prop = str(record.get("source_prop_policy", "unspecified"))
        reason = ("source primary present" if inject else
                  "source primary absent; secondary human retained" if policy["allow_secondary_humans"] else
                  "strict empty: primary and secondary humans absent")
        uses = ["used" if ref is not None else "omitted" for ref in refs]
        report = (f"shot={number} seed={int(seed)} subject_mode={mode} secondary_human_presence={secondary} "
                  f"source_prop_policy={prop} strict_empty={bool(policy['strict_empty'])} "
                  f"picture1={uses[0]} picture2={uses[1]} picture3={uses[2]} reason={reason}")
        return (*refs, bool(policy["strict_empty"]), bool(policy["allow_secondary_humans"]),
                inject, mode, secondary, prop, reason, report, _has_reference_tensor(picture3),
                *(ref is not None for ref in refs), int(seed))


class H3CPromptCompiler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_prompts": ("STRING", {"forceInput": True}),
            "shot_number": ("INT", {"forceInput": True}),
            "prompt_first": ("STRING", {"multiline": True, "default": ""}),
            "prompt_second": ("STRING", {"multiline": True, "default": ""}),
            "strict_empty": ("BOOLEAN", {"forceInput": True}),
            "allow_secondary_humans": ("BOOLEAN", {"forceInput": True}),
            "picture3_present": ("BOOLEAN", {"forceInput": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("first_pass_prompt", "second_pass_prompt", "scene_fields", "filtered_fields",
                    "picture3_contract", "reference_policy", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, shot_prompts, shot_number, prompt_first, prompt_second, strict_empty,
            allow_secondary_humans, picture3_present):
        rows = _parse_scene_prompts(shot_prompts)
        number = int(shot_number)
        if number not in rows:
            raise ValueError(f"shot {number} is missing from scene records")
        policy = "strict_empty" if strict_empty else "secondary_only" if allow_secondary_humans else "target_present"
        first = _compose_shot_prompt(rows, number - 1, prompt_first, bool(strict_empty), "first",
                                     allow_secondary_humans=bool(allow_secondary_humans),
                                     picture3_present=bool(picture3_present))
        second = _compose_shot_prompt(rows, number - 1, prompt_second, bool(strict_empty), "second",
                                      allow_secondary_humans=bool(allow_secondary_humans),
                                      picture3_present=bool(picture3_present))
        scene_fields = json.dumps(rows[number].get("scene_fields", {}), ensure_ascii=False, sort_keys=True)
        # Equivalent-to-old-C mode: the old normaliser did not semantically filter scene fields.
        filtered_fields = "[]"
        contract = ("not_applicable" if policy != "target_present" else
                    "included" if picture3_present else "removed")
        report = f"shot={number} reference_policy={policy} picture3_contract={contract}\nFIRST PASS:\n{first}\nSECOND PASS:\n{second}"
        return first, second, scene_fields, filtered_fields, contract, policy, report


class H3CShotTrim:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"decoded": ("IMAGE",), "original_f": ("INT", {"forceInput": True}),
                             "work_l": ("INT", {"forceInput": True}),
                             "shot_number": ("INT", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("trimmed", "frame_count", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, decoded, original_f, work_l, shot_number):
        count = int(original_f)
        if count <= 0 or count > int(work_l) or int(decoded.shape[0]) != int(work_l):
            raise ValueError(f"shot {shot_number}: decoded/work_l/original_f frame contract mismatch")
        trimmed = decoded[:count].detach().to(device="cpu", dtype=torch.float32)
        if trimmed.ndim != 4:
            raise ValueError(f"shot {shot_number}: decoded IMAGE must be [T,H,W,C]")
        return trimmed, count, f"shot={int(shot_number)} trimmed={count} decoded={int(decoded.shape[0])} dtype=float32 device=cpu"


class H3COrderedMergeStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"previous": ("*",), "shot_frames": ("IMAGE",),
                             "shot_number": ("INT", {"forceInput": True}),
                             "total_frames": ("INT", {"forceInput": True}),
                             "shot_range": ("STRING", {"forceInput": True}),
                             "policy_report": ("STRING", {"forceInput": True}),
                             "prompt_report": ("STRING", {"forceInput": True}),
                             "shot_report": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("state", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, previous, shot_frames, shot_number, total_frames, shot_range,
            policy_report, prompt_report, shot_report):
        total = int(total_frames)
        state = previous if previous is not None else {
            "next_shot": 1, "total_frames": total, "frame_count": 0,
            "shape": tuple(shot_frames.shape[1:]), "chunks": [], "reports": [],
        }
        if not isinstance(state, dict) or int(state["next_shot"]) != int(shot_number):
            raise ValueError("shot merge order mismatch")
        if int(state["total_frames"]) != total or tuple(state["shape"]) != tuple(shot_frames.shape[1:]):
            raise ValueError("shot merge total/shape mismatch")
        frame_count = int(state["frame_count"]) + int(shot_frames.shape[0])
        if frame_count > total:
            raise ValueError("shot merge exceeds total frame count")
        shot_diagnostic = (f"shot={int(shot_number)} range={shot_range}\n{policy_report}\n"
                           f"{prompt_report}\n{shot_report}")
        result = {**state, "next_shot": int(shot_number) + 1, "frame_count": frame_count,
                  "chunks": [*state["chunks"], shot_frames],
                  "reports": [*state["reports"], shot_diagnostic]}
        return result, f"merged_shot={shot_number} frames={frame_count}/{total}"


class H3COrderedMergeFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"state": ("*",), "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                             "total_frames": ("INT", {"forceInput": True})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "fps", "frame_count", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, state, fps, total_frames):
        total = int(total_frames)
        if not isinstance(state, dict) or not state.get("chunks") or int(state["frame_count"]) != total:
            raise ValueError("ordered merge is incomplete")
        if int(state["total_frames"]) != total:
            raise ValueError("ordered merge total frame count changed")
        output = torch.empty((total, *tuple(state["shape"])), dtype=torch.float32, device="cpu")
        cursor = 0
        for chunk in state["chunks"]:
            count = int(chunk.shape[0])
            output[cursor:cursor + count].copy_(chunk)
            cursor += count
        if cursor != total:
            raise ValueError("ordered merge copied an unexpected frame count")
        report = "H3 C modular ordered merge\n" + "\n".join(state["reports"])
        return output, float(fps), total, report


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3CShotPlanner, H3CShotSelectPad, H3CShotPolicy, H3CPromptCompiler,
    H3CShotTrim, H3COrderedMergeStep, H3COrderedMergeFinish,
)}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CShotPlanner": "H3 C Shot Planner / Manifest",
    "H3CShotSelectPad": "H3 C Shot Select / Depth Pad",
    "H3CShotPolicy": "H3 C Shot / Reference Policy",
    "H3CPromptCompiler": "H3 C Prompt Compiler",
    "H3CShotTrim": "H3 C Shot Trim",
    "H3COrderedMergeStep": "H3 C Ordered Merge Step",
    "H3COrderedMergeFinish": "H3 C Ordered Merge Finish",
}
