"""Independent H3 Runner C: shot-local scene VLM plus optional Picture 3.

This module deliberately leaves the formal runner and the A/B candidates
untouched.  The scene node owns only deterministic shot-aligned VLM analysis;
Runner C reuses the proven A execution sequence and adds an optional third
reference image for a back-garment view.
"""

from __future__ import annotations

import json
import inspect
import re
from typing import Any

import torch

try:
    import nodes
except Exception:  # pragma: no cover - imported inside ComfyUI
    nodes = None

from .h3_hardcut_runner_a import (
    _call_node,
    _compose_shot_prompt as _compose_base_shot_prompt,
    _derive_shot_seed,
    _detect_hard_cuts,
    _fit_length,
    _legal_length,
    _parse_empty_shots,
    _parse_scene_prompts as _parse_base_scene_prompts,
    _sample,
    _unwrap,
    _upscale_video_latent,
    _validate_scene_prompts,
)
from .h3_hardcut_runner_b import _apply_h3_fun_control


_SCENE_VLM_PROMPT = """You are a strict shot-local video scene analyst.
Analyze only the supplied RGB frames from this one shot. Return exactly one
JSON object and no markdown. Use this schema:
{
    "subject_mode": "present",
    "secondary_human_presence": "none",
  "depicted_human_like_objects": "posters, photographs, screen images, advertisements, mannequins, statues, or other human-like depictions/objects; never classify as secondary humans",
  "scene": "factual description of the real environment",
  "space_structure": "architecture, layout, depth and spatial relations",
  "environment_objects": "only stable background/environment objects",
  "foreground_midground_background": "what occupies each depth layer",
  "materials_colors": "visible materials and true environment colors",
  "lighting": "light direction, exposure, shadows, highlights, contrast",
  "color_temperature_palette": "cool/warm relations and palette",
  "composition_camera": "shot scale, viewpoint, framing and camera movement",
  "scene_motion": "camera movement and non-identity environmental motion",
  "temporal_structure": "timing, pacing and non-identity temporal transitions",
  "expression_gaze_mouth": "only visible expression, gaze and mouth state; use unknown for absent shots",
    "subject_associated_prop_policy": "none"
  }
  subject_mode must be present or absent. secondary_human_presence must be none,
  present, or uncertain. subject_associated_prop_policy must be none, present,
  absent, or uncertain. The secondary-human field is only for real
humans physically present in the shot or their real mirror/glass reflections.
Set subject_mode=present only when the source primary protagonist is visible or
identifiable in this shot. It is not a count of every human-shaped signal.
secondary_human_presence is only for real human beings physically present in
the shot or real human reflections in a mirror or glass. Background pedestrians
and distant real people may be present; a distant or occluded figure may be
uncertain only when there is evidence that it is a real human. Posters,
photographs, screen images, advertisements, mannequins, statues, and other
human-like depictions or objects are not humans for this field: set
secondary_human_presence=none and record them only in environment_objects or
depicted_human_like_objects. Never use secondary_human_presence to inject the
target-person references.
Do not describe or infer the source person's identity, face, hair, body, clothing,
clothing color, clothing material, or action choreography. A handheld item is
subject-associated, not a fixed background object. A source-only temporary prop
or accessory must never become target identity or appearance; report a visible
handheld item only through subject_associated_prop_policy. Do not invent objects
that are not visible. For an empty shot set subject_mode=absent and describe
only the environment, camera and atmosphere."""

_SCENE_FIELDS = (
    ("scene", "Scene"),
    ("space_structure", "Space structure"),
    ("environment_objects", "Environment objects"),
    ("depicted_human_like_objects", "Depicted human-like objects"),
    ("foreground_midground_background", "Foreground/midground/background"),
    ("materials_colors", "Materials and environment colors"),
    ("lighting", "Lighting"),
    ("color_temperature_palette", "Color temperature and palette"),
    ("composition_camera", "Composition and camera"),
    ("scene_motion", "Scene motion"),
    ("temporal_structure", "Temporal structure"),
)


def _available_vlm_models() -> list[str]:
    """Read the installed Qwen-VL model catalog when the node is available."""
    fallback = [
        "Qwen3-VL-32B-Instruct",
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-4B-Instruct",
    ]
    if nodes is None:
        return fallback
    for name in ("AILab_QwenVL_Advanced", "AILab_QwenVL"):
        cls = nodes.NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            continue
        try:
            spec = cls.INPUT_TYPES()
            values = spec.get("required", {}).get("model_name", ([],))[0]
            values = [str(value) for value in values if str(value).strip()]
            if values:
                preferred = [
                    "Qwen3-VL-32B-Instruct",
                    "Qwen3-VL-8B-Instruct",
                    "Qwen3-VL-4B-Instruct",
                ]
                ordered = [item for item in preferred if item in values]
                ordered.extend(item for item in values if item not in ordered)
                return ordered
        except Exception:
            continue
    return fallback


def _clean_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{") or not text.endswith("}"):
        raise ValueError("per-shot VLM must return exactly one JSON object")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"per-shot VLM returned malformed JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("per-shot VLM JSON must be an object")
    return value


def _text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return "; ".join(f"{key}: {val}" for key, val in value.items())
    return str(value or "").strip()


def _normalise_scene_record(raw: str, shot_id: int) -> dict[str, Any]:
    obj = _clean_json_object(raw)
    mode = _text(obj.get("subject_mode")).lower()
    if mode not in {"present", "absent"}:
        raise ValueError(f"shot {shot_id}: subject_mode must be present or absent")

    secondary_human_presence = _text(obj.get("secondary_human_presence")).lower()
    if secondary_human_presence not in {"none", "present", "uncertain"}:
        secondary_human_presence = "uncertain"

    parts: list[str] = []
    scene_fields: dict[str, str] = {}
    for key, label in _SCENE_FIELDS:
        value = _text(obj.get(key))
        if value:
            scene_fields[key] = value
            parts.append(f"{label}: {value}")
    if not parts:
        fallback = _text(obj.get("prompt"))
        if fallback:
            parts.append(fallback)
    if not parts:
        raise ValueError(f"shot {shot_id}: VLM returned no scene fields")

    if mode == "present":
        expression = _text(obj.get("expression_gaze_mouth"))
        if expression and expression.lower() not in {"unknown", "n/a", "none"}:
            parts.append(f"Visible expression/gaze/mouth state: {expression}")
        if secondary_human_presence in {"present", "uncertain"}:
            parts.append(
                "Secondary-human policy: preserve source-visible secondary/background humans "
                "as separate non-target people without deleting or replacing them."
            )
    elif secondary_human_presence in {"present", "uncertain"}:
        parts.insert(
            0,
            "No person is visible as the source primary protagonist; preserve only source-visible "
            "secondary/background humans without injecting the target identity.",
        )
    else:
        parts.insert(0, "No person is visible in this shot; preserve only the environment, camera and atmosphere.")

    prop_policy = _text(
        obj.get("subject_associated_prop_policy", obj.get("subject_associated_prop"))
    ).lower()
    if prop_policy in {"present", "absent", "uncertain", "none"}:
        parts.append(
            "Subject-associated prop policy: "
            f"{prop_policy}; do not turn a handheld prop into a fixed environment object."
        )

    record: dict[str, Any] = {
        "prompt": "\n".join(parts),
        "subject_mode": mode,
        "secondary_human_presence": secondary_human_presence,
        "scene_fields": scene_fields,
    }
    if prop_policy in {"present", "absent", "uncertain", "none"}:
        record["source_prop_policy"] = prop_policy
    return record


def _parse_scene_prompts(value: str) -> dict[int, dict[str, Any]]:
    """Parse A-compatible shot prompts while retaining C/D scene metadata."""
    rows = _parse_base_scene_prompts(value)
    lines = [raw.strip() for raw in str(value or "").splitlines() if raw.strip()]
    structured = any(line.startswith("{") for line in lines)
    if not structured:
        for record in rows.values():
            record.setdefault("secondary_human_presence", "none")
        return rows
    for line in lines:
        obj = json.loads(line)
        shot = int(obj["shot"])
        record = rows[shot]
        if "secondary_human_presence" not in obj:
            # Preserve the legacy JSONL contract: old A-only records have no
            # secondary-human field and therefore remain strict-empty rows.
            secondary = "none"
        else:
            secondary = _text(obj.get("secondary_human_presence")).lower()
            if secondary not in {"none", "present", "uncertain"}:
                secondary = "uncertain"
        record["secondary_human_presence"] = secondary
        prop = _text(obj.get("source_prop_policy", obj.get("subject_associated_prop_policy"))).lower()
        if prop in {"none", "present", "absent", "uncertain"}:
            record["source_prop_policy"] = prop
    return rows


def _resolve_shot_policy(scene_record, shot_number, manual_empty_shots):
    """Resolve source-protagonist, secondary-human, and strict-empty behavior."""
    mode = str(scene_record.get("subject_mode", "present")).strip().lower()
    secondary = str(scene_record.get("secondary_human_presence", "none")).strip().lower()
    manual = int(shot_number) in manual_empty_shots
    if manual and mode == "present":
        raise ValueError(
            f"empty_shot_indices conflicts with subject_mode=present for shot {int(shot_number)}"
        )
    if manual and secondary in {"present", "uncertain"}:
        raise ValueError(
            "empty_shot_indices conflicts with "
            f"secondary_human_presence={secondary!r} for shot {int(shot_number)}"
        )
    if manual:
        return {"strict_empty": True, "allow_secondary_humans": False, "inject_target_references": False}
    if mode == "present":
        return {"strict_empty": False, "allow_secondary_humans": False, "inject_target_references": True}
    if secondary in {"present", "uncertain"}:
        return {"strict_empty": False, "allow_secondary_humans": True, "inject_target_references": False}
    return {"strict_empty": True, "allow_secondary_humans": False, "inject_target_references": False}


def _resolve_empty_shot(scene_record, shot_number, manual_empty_shots):
    return bool(_resolve_shot_policy(scene_record, shot_number, manual_empty_shots)["strict_empty"])


def _parse_manual_force_empty_indices(value: str, shot_count: int | None = None) -> set[int]:
    """Parse the user-facing force-empty override independently of auto output."""
    try:
        return _parse_empty_shots(value, shot_count=shot_count)
    except ValueError as exc:
        raise ValueError(f"manual_force_empty_indices: {exc}") from exc


def _has_reference_tensor(value) -> bool:
    if value is None:
        return False
    try:
        return int(value.shape[0]) > 0
    except (AttributeError, IndexError, TypeError, ValueError):
        return True


_PICTURE3_CONTRACT_BEGIN = "[[PICTURE3_CONTRACT_BEGIN]]"
_PICTURE3_CONTRACT_END = "[[PICTURE3_CONTRACT_END]]"
_PICTURE3_CONTRACT_BLOCK = re.compile(
    re.escape(_PICTURE3_CONTRACT_BEGIN)
    + r"(?P<body>.*?)"
    + re.escape(_PICTURE3_CONTRACT_END),
    flags=re.DOTALL,
)


def _legacy_without_picture3_contract(prompt: str) -> str:
    """Conservatively clean legacy unmarked Picture3 prose."""
    text = str(prompt or "")
    picture3_token = re.compile(
        r"(?<![A-Za-z0-9_])(?:<\s*Picture\s*3\s*>|Picture\s*3)(?!\d)",
        flags=re.IGNORECASE,
    )
    picture12_token = re.compile(
        r"(?<![A-Za-z0-9_])(?:<\s*Picture\s*[12]\s*>|Picture\s*[12])(?!\d)",
        flags=re.IGNORECASE,
    )
    conjunction = re.compile(r"(?i)\b(?:and|or)\b|[\u548c\u4e0e\u4ee5\u53ca]+")
    preserved: list[str] = []
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?\u3002\uFF01\uFF1F])", line):
            sentence = sentence.strip()
            if not sentence:
                continue
            terminal = sentence[-1] if sentence[-1] in ".!?\u3002\uFF01\uFF1F" else ""
            body = sentence[:-1] if terminal else sentence
            fragments: list[str] = []
            for clause in re.split(r"[,\uFF0C;\uFF1B]", body):
                clause = clause.strip()
                if not clause:
                    continue
                fragments.extend(part.strip() for part in conjunction.split(clause) if part.strip())
            if not any(picture3_token.search(part) for part in fragments):
                if body.strip():
                    preserved.append(body.strip() + terminal)
                continue

            kinds = []
            for part in fragments:
                has_picture3 = bool(picture3_token.search(part))
                has_picture12 = bool(picture12_token.search(part))
                kinds.append((has_picture3, has_picture12))

            # An unmarked sentence with a non-reference fragment between the
            # optional reference and Picture1/2 is structurally ambiguous.
            # Dropping the whole sentence is safer than leaving a dangling
            # pronoun or conditional remainder.  Production prompts use the
            # explicit block contract below instead of this legacy fallback.
            for left, (has3, has12) in enumerate(kinds):
                if not has3 or has12:
                    continue
                for right in range(left + 1, len(kinds)):
                    right_has3, right_has12 = kinds[right]
                    if right_has12 and not right_has3 and any(
                        not kind3 and not kind12 for kind3, kind12 in kinds[left + 1 : right]
                    ):
                        fragments = []
                        break
                if not fragments:
                    break

            if not fragments:
                continue

            kept: list[str] = []
            for part, (has_picture3, has_picture12) in zip(fragments, kinds):
                if not has_picture3:
                    kept.append(part)
                elif has_picture12:
                    # Both ordinals occur in one unmarked fragment.  There is
                    # no reliable natural-language boundary to rewrite safely.
                    continue
            if kept:
                preserved.append(", ".join(kept) + terminal)
    return re.sub(r"\s+", " ", " ".join(preserved)).strip()


def _apply_picture3_contract(prompt: str, *, picture3_present: bool) -> str:
    """Apply explicit Picture3 blocks, with a conservative legacy fallback."""
    text = str(prompt or "")

    def replace_block(match: re.Match) -> str:
        return match.group("body") if picture3_present else ""

    text = _PICTURE3_CONTRACT_BLOCK.sub(replace_block, text)
    text = text.replace(_PICTURE3_CONTRACT_BEGIN, "").replace(_PICTURE3_CONTRACT_END, "")
    if picture3_present:
        return re.sub(r"\s+", " ", text).strip()
    return _legacy_without_picture3_contract(text)


def _without_picture3_contract(prompt: str) -> str:
    """Remove Picture3 blocks or conservatively clean legacy prose."""
    return _apply_picture3_contract(prompt, picture3_present=False)


def _compose_shot_prompt(
    scene_prompts,
    shot_id,
    global_prompt,
    is_empty,
    pass_name,
    *,
    allow_secondary_humans=False,
    picture3_present=False,
):
    scene_record = scene_prompts[shot_id + 1]
    scene = scene_record["prompt"]
    if is_empty:
        return (
            f"{scene}\n"
              "Strict-empty policy: no real person and no target identity reference. "
              "Preserve all source-visible environment and depicted human-like objects "
              "including posters, photographs, screen images, advertisements, mannequins "
              "and statues as scene elements only when actually visible in the supplied "
              "source shot; never invent objects absent from the source shot, do not turn "
              "them into real humans or inject "
            f"target references. This is the {pass_name} pass."
        )
    if allow_secondary_humans:
        return (
            f"{scene}\n"
            "Secondary-human policy: source primary protagonist absent; preserve only source-visible "
            "background, reflected, distant, or incidental humans. Preserve the scene, "
            "spatial relations, camera, environmental motion, and temporal structure "
            "described above. Never inject target identity, target appearance, target "
            "references, or invent a source primary protagonist."
        )
    prompt = _apply_picture3_contract(str(global_prompt or ""), picture3_present=picture3_present)
    return _compose_base_shot_prompt(scene_prompts, shot_id, prompt, bool(is_empty), pass_name)


def _scene_vlm_values(
    shot_rgb: torch.Tensor,
    *,
    model_name: str,
    quantization: str,
    attention_mode: str,
    use_torch_compile: bool,
    device: str,
    max_tokens: int,
    frame_count: int,
    video_frame_size: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "model_name": str(model_name),
        "quantization": str(quantization),
        "attention_mode": str(attention_mode),
        "use_torch_compile": bool(use_torch_compile),
        "device": str(device),
        "preset_prompt": "🖼️ Detailed Description",
        "custom_prompt": _SCENE_VLM_PROMPT,
        "max_tokens": int(max_tokens),
        "temperature": 0.2,
        "top_p": 0.9,
        "num_beams": 1,
        "repetition_penalty": 1.1,
        "frame_count": max(1, int(frame_count)),
        "video_frame_size": str(video_frame_size),
        "keep_model_loaded": True,
        "seed": int(seed),
        "image": None,
        "video": shot_rgb,
    }


class _SceneVLMInvoker:
    """Reuse one registered Qwen node instance for all shots in one run.

    ComfyUI normally caches a node object for the lifetime of a workflow
    execution.  Runner C calls the VLM from Python, so using the shared
    ``_call_node`` helper would otherwise construct a fresh object per shot
    and defeat instance-local ``keep_model_loaded`` caches.
    """

    def __init__(self):
        if nodes is None:
            raise RuntimeError("H3ShotSceneVLM requires AILab_QwenVL_Advanced or AILab_QwenVL")
        if "AILab_QwenVL_Advanced" in nodes.NODE_CLASS_MAPPINGS:
            class_type = "AILab_QwenVL_Advanced"
        elif "AILab_QwenVL" in nodes.NODE_CLASS_MAPPINGS:
            class_type = "AILab_QwenVL"
        else:
            raise RuntimeError("H3ShotSceneVLM requires AILab_QwenVL_Advanced or AILab_QwenVL")
        cls = nodes.NODE_CLASS_MAPPINGS[class_type]
        self.class_type = class_type
        self.instance = cls()
        self.function_name = getattr(cls, "FUNCTION", "execute")
        self.function = getattr(self.instance, self.function_name)
        signature = inspect.signature(self.function)
        if self.function_name in {"EXECUTE_NORMALIZED", "EXECUTE_NORMALIZED_ASYNC"}:
            execute_fn = getattr(cls, "execute", None)
            if execute_fn is not None:
                signature = inspect.signature(execute_fn)
        self.accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        self.parameters = set(signature.parameters)
        self.effective_history: list[dict[str, Any]] = []
        self.filtered_history: list[list[str]] = []

    def __call__(
        self,
        values: dict[str, Any],
        *,
        configured_frame_count: int | None = None,
        shot_frame_count: int | None = None,
    ) -> tuple:
        call_values = dict(values) if self.accepts_kwargs else {
            key: value for key, value in values.items() if key in self.parameters
        }
        wrapper_requested = int(values.get("frame_count", 0))
        requested_settings = {
            "frame_count": wrapper_requested,
            "temperature": values.get("temperature"),
            "top_p": values.get("top_p"),
            "num_beams": values.get("num_beams"),
            "repetition_penalty": values.get("repetition_penalty"),
            "video_frame_size": values.get("video_frame_size"),
            "device": values.get("device"),
            "requested_device": values.get("device"),
            "use_torch_compile": values.get("use_torch_compile"),
        }
        effective = dict(values)
        if self.class_type == "AILab_QwenVL":
            # Standard AILab_QwenVL has no advanced inputs and its process()
            # wrapper calls run() with these fixed upstream values.
            effective = {
                "frame_count": 16,
                "temperature": 0.6,
                "top_p": 0.9,
                "num_beams": 1,
                "repetition_penalty": 1.2,
                "video_frame_size": "auto",
                "device": "unknown / not verified",
                "requested_device": values.get("device"),
                "node_device_argument": "auto",
                "runtime_resolved_device": "unknown / not verified",
                "use_torch_compile": False,
            }
            node_requested = 16
            node_internal_limit = 16
        else:
            node_requested = wrapper_requested
            node_internal_limit = "not_fixed"
            effective = {
                "frame_count": node_requested,
                "temperature": values.get("temperature"),
                "top_p": values.get("top_p"),
                "num_beams": values.get("num_beams"),
                "repetition_penalty": values.get("repetition_penalty"),
                "video_frame_size": values.get("video_frame_size"),
                # The upstream node resolves the argument against live device
                # availability; this wrapper cannot observe that runtime result.
                "device": "unknown / not verified",
                "requested_device": values.get("device"),
                "node_device_argument": values.get("device"),
                "runtime_resolved_device": "unknown / not verified",
                "use_torch_compile": (
                    False if not values.get("use_torch_compile") else "unknown"
                ),
            }
        actual_sampled = None
        if shot_frame_count is not None:
            actual_sampled = min(max(int(shot_frame_count), 0), int(node_requested))
        filtered = sorted(key for key in values if key not in call_values)
        self.effective_history.append(
            {
                "configured_frame_count": (
                    None if configured_frame_count is None else int(configured_frame_count)
                ),
                "shot_frame_count": (
                    None if shot_frame_count is None else int(shot_frame_count)
                ),
                "requested_frame_count": wrapper_requested,
                "wrapper_requested_frame_count": wrapper_requested,
                "node_requested_frame_count": int(node_requested),
                "node_internal_limit": node_internal_limit,
                "actual_sampled_frame_count": actual_sampled,
                # Keep the original diagnostic key for local callers/tests while
                # exposing the richer requested/effective settings below.
                "frame_count": int(effective["frame_count"]),
                "effective_frame_count": (
                    actual_sampled
                    if actual_sampled is not None
                    else int(effective["frame_count"])
                ),
                "node_effective_frame_count": int(effective["frame_count"]),
                "requested_settings": requested_settings,
                "effective_settings": effective,
                "filtered_parameters": filtered,
            }
        )
        self.filtered_history.append(filtered)
        return _unwrap(self.function(**call_values))

    def report_snapshot(self) -> dict[str, Any]:
        latest = self.effective_history[-1] if self.effective_history else {}
        return {"node_class": self.class_type, **latest, "shots": list(self.effective_history)}


def _call_scene_vlm(
    shot_rgb: torch.Tensor,
    *,
    model_name: str,
    quantization: str,
    attention_mode: str,
    use_torch_compile: bool,
    device: str,
    max_tokens: int,
    frame_count: int,
    video_frame_size: str,
    seed: int,
    invoker: _SceneVLMInvoker | None = None,
    configured_frame_count: int | None = None,
    shot_frame_count: int | None = None,
) -> str:
    values = _scene_vlm_values(
        shot_rgb,
        model_name=model_name,
        quantization=quantization,
        attention_mode=attention_mode,
        use_torch_compile=use_torch_compile,
        device=device,
        max_tokens=max_tokens,
        frame_count=frame_count,
        video_frame_size=video_frame_size,
        seed=seed,
    )
    active_invoker = invoker or _SceneVLMInvoker()
    if isinstance(active_invoker, _SceneVLMInvoker):
        out = active_invoker(
            values,
            configured_frame_count=configured_frame_count,
            shot_frame_count=shot_frame_count,
        )
    else:
        out = active_invoker(values)
    if not out or not str(out[0]).strip():
        raise RuntimeError("per-shot VLM returned an empty response")
    return str(out[0])


class H3ShotSceneVLM:
    """Detect hard cuts and create one strict scene record per detected shot."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _available_vlm_models()
        return {
            "required": {
                "rgb": ("IMAGE",),
                "model_name": (models, {"default": models[0]}),
                "quantization": (["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"], {"default": "4-bit (VRAM-friendly)"}),
                "attention_mode": (["auto", "sdpa", "flash_attention_2", "sage"], {"default": "auto"}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cpu", "mps", "cuda:0"], {"default": "auto"}),
                "max_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 64}),
                "frame_count": ("INT", {"default": 32, "min": 1, "max": 64}),
                "video_frame_size": (["auto", "384", "448", "512", "768", "original"], {"default": "auto"}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "cut_threshold": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_shot_frames": ("INT", {"default": 8, "min": 1, "max": 240}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("shot_prompts", "empty_shot_indices", "report")
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3"

    def run(
        self,
        rgb,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        max_tokens,
        frame_count,
        video_frame_size,
        seed,
        cut_threshold,
        min_shot_frames,
    ):
        shots = _detect_hard_cuts(rgb, cut_threshold, min_shot_frames)
        rows: list[str] = []
        empty: list[str] = []
        report = [
            "H3ShotSceneVLM: deterministic shot-local analysis",
            f"shots={len(shots)}, threshold={float(cut_threshold):.6g}, min_shot_frames={int(min_shot_frames)}",
            f"model={model_name}, configured_frame_count={int(frame_count)}, image_reference=DISABLED",
        ]
        scene_vlm = _SceneVLMInvoker()
        for shot_index, (start, end) in enumerate(shots, 1):
            shot_rgb = rgb[start:end]
            response = _call_scene_vlm(
                shot_rgb,
                model_name=str(model_name),
                quantization=str(quantization),
                attention_mode=str(attention_mode),
                use_torch_compile=bool(use_torch_compile),
                device=str(device),
                max_tokens=int(max_tokens),
                frame_count=min(int(frame_count), max(1, int(end - start))),
                video_frame_size=str(video_frame_size),
                seed=int(seed) + shot_index,
                invoker=scene_vlm,
                configured_frame_count=int(frame_count),
                shot_frame_count=int(end - start),
            )
            record = _normalise_scene_record(response, shot_index)
            record["shot"] = shot_index
            rows.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            if (
                record["subject_mode"] == "absent"
                and record["secondary_human_presence"] == "none"
            ):
                empty.append(str(shot_index))
            report.append(
                f"shot={shot_index} range=[{start},{end}) frames={end-start} "
                f"subject={record['subject_mode']} "
                f"secondary_humans={record['secondary_human_presence']} "
                f"strict_empty={record['subject_mode'] == 'absent' and record['secondary_human_presence'] == 'none'} "
                f"source_prop={record.get('source_prop_policy', 'unspecified')} "
                f"scene_metadata={json.dumps(record['scene_fields'], ensure_ascii=False, separators=(',', ':'))}"
            )
        if len(rows) != len(shots):
            raise RuntimeError("VLM/prompt shot count mismatch")
        report.append("vlm_runtime=" + json.dumps(scene_vlm.report_snapshot(), ensure_ascii=False, sort_keys=True))
        return ("\n".join(rows), ",".join(empty), "\n".join(report))


class H3OptionalPicture3:
    """Native upload widget that resolves to IMAGE or a real None value."""

    @classmethod
    def INPUT_TYPES(cls):
        load_image = getattr(nodes, "LoadImage", None) if nodes is not None else None
        source = None
        if load_image is not None:
            try:
                source = load_image.INPUT_TYPES().get("required", {}).get("image")
            except Exception:
                source = None
        options = list(source[0]) if source and isinstance(source[0], (list, tuple)) else []
        if "" not in options:
            options.insert(0, "")
        config = dict(source[1]) if source and len(source) > 1 and isinstance(source[1], dict) else {}
        config["image_upload"] = True
        return {"required": {"image": (options, config)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load"
    CATEGORY = "video/MiniMaxH3"

    def load(self, image):
        selected = str(image or "").strip()
        if not selected:
            return (None,)
        if nodes is None or not hasattr(nodes, "LoadImage"):
            raise RuntimeError("ComfyUI LoadImage is required for H3OptionalPicture3")
        result = nodes.LoadImage().load_image(selected)
        if not result or result[0] is None:
            return (None,)
        return (result[0],)

    @classmethod
    def IS_CHANGED(cls, image):
        selected = str(image or "").strip()
        if not selected:
            return "H3OptionalPicture3:empty"
        if nodes is None or not hasattr(nodes, "LoadImage"):
            raise RuntimeError("ComfyUI LoadImage is required for H3OptionalPicture3")
        checker = getattr(nodes.LoadImage, "IS_CHANGED", None)
        if checker is None:
            raise RuntimeError("ComfyUI LoadImage.IS_CHANGED is required for H3OptionalPicture3")
        return checker(selected)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        selected = str(image or "").strip()
        if not selected:
            return True
        if nodes is None or not hasattr(nodes, "LoadImage"):
            raise RuntimeError("ComfyUI LoadImage is required for H3OptionalPicture3")
        validator = getattr(nodes.LoadImage, "VALIDATE_INPUTS", None)
        if validator is None:
            raise RuntimeError("ComfyUI LoadImage.VALIDATE_INPUTS is required for H3OptionalPicture3")
        return validator(selected)


def _h3_ref2v_with_picture3(
    *,
    clip,
    vae,
    prompt: str,
    width: int,
    height: int,
    length: int,
    picture1: torch.Tensor | None,
    picture2: torch.Tensor | None,
    picture3: torch.Tensor | None,
    depth: torch.Tensor | None,
):
    refs = {}
    for index, picture in enumerate((picture1, picture2, picture3)):
        if picture is not None and int(picture.shape[0]) > 0:
            refs[f"ref_image_{index}"] = picture[:1]
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
            **refs,
            **videos,
        },
    )
    if len(out) < 2:
        raise RuntimeError("MiniMaxH3ReferenceToVideo did not return conditioning + latent")
    return out[0], out[1]


def _picture_refs_report(inject_target_references, picture1, picture2, picture3) -> str:
    if not inject_target_references:
        return "picture1=omitted,picture2=omitted,picture3=omitted"
    return ",".join(
        f"picture{index}={'used' if _has_reference_tensor(picture) else 'omitted'}"
        for index, picture in enumerate((picture1, picture2, picture3), 1)
    )


class H3HardCutRunnerSceneVLM:
    """Runner C: A's stable generation chain plus optional Picture 3."""

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
            },
            "optional": {
                "manual_force_empty_indices": ("STRING", {"default": ""}),
                "picture3": ("IMAGE",),
            },
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
        manual_force_empty_indices="",
        picture3=None,
    ):
        if int(rgb.shape[0]) != int(depth.shape[0]):
            raise ValueError(
                f"RGB/Depth frame count mismatch: {int(rgb.shape[0])} vs {int(depth.shape[0])}"
            )
        shots = _detect_hard_cuts(rgb, cut_threshold, min_shot_frames)
        scene_prompts = _parse_scene_prompts(shot_prompts)
        empty_shots = _parse_empty_shots(empty_shot_indices, shot_count=len(shots))
        empty_shots |= _parse_manual_force_empty_indices(
            manual_force_empty_indices, shot_count=len(shots)
        )
        _validate_scene_prompts(scene_prompts, len(shots))
        total_frames = sum(int(end - start) for start, end in shots)
        output_images: torch.Tensor | None = None
        output_cursor = 0
        report_lines = [
            "H3HardCutRunner C: shot-local VLM + optional Picture 3",
            f"shots={len(shots)}, fps={float(fps):.6g}, threshold={float(cut_threshold):.4g}",
            f"picture3={'present' if picture3 is not None else 'omitted'}",
            f"manual_force_empty_indices={str(manual_force_empty_indices or '').strip() or 'none'}",
        ]
        for shot_id, (start, end) in enumerate(shots):
            original_f = int(end - start)
            scene_record = scene_prompts[shot_id + 1]
            policy = _resolve_shot_policy(scene_record, shot_id + 1, empty_shots)
            is_empty = bool(policy["strict_empty"])
            allow_secondary_humans = bool(policy["allow_secondary_humans"])
            inject_target_references = bool(policy["inject_target_references"])
            shot_seed = _derive_shot_seed(seed, shot_id)
            work_l = _legal_length(original_f)
            depth_shot = _fit_length(depth[start:end], work_l)
            first_positive, first_latent = _h3_ref2v_with_picture3(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(
                    scene_prompts, shot_id, str(prompt_first or ""), is_empty, "first",
                    allow_secondary_humans=allow_secondary_humans,
                    picture3_present=_has_reference_tensor(picture3),
                ),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=picture1 if inject_target_references else None,
                picture2=picture2 if inject_target_references else None,
                picture3=picture3 if inject_target_references else None,
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
                "LTXVConcatAVLatent", {"video_latent": upscaled_video, "audio_latent": split[1]}
            )[0]
            second_positive, _unused_latent = _h3_ref2v_with_picture3(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(
                    scene_prompts, shot_id, str(prompt_second or ""), is_empty, "second",
                    allow_secondary_humans=allow_secondary_humans,
                    picture3_present=_has_reference_tensor(picture3),
                ),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=picture1 if inject_target_references else None,
                picture2=picture2 if inject_target_references else None,
                picture3=picture3 if inject_target_references else None,
                depth=None,
            )
            second_output, _second_clean = _sample(
                model, second_positive, sampler, low_sigmas, joined, shot_seed
            )
            decoded = _call_node("VAEDecode", {"samples": second_output, "vae": video_vae})[0]
            trimmed = decoded[:original_f].detach().to(device="cpu", dtype=torch.float32)
            if output_images is None:
                output_images = torch.empty(
                    (total_frames, *tuple(trimmed.shape[1:])), dtype=trimmed.dtype, device="cpu"
                )
            elif tuple(output_images.shape[1:]) != tuple(trimmed.shape[1:]):
                raise RuntimeError("Shot decode shape changed across hard-cut segments")
            output_images[output_cursor : output_cursor + original_f].copy_(trimmed)
            output_cursor += original_f
            report_lines.append(
                f"shot={shot_id + 1} range=[{start},{end}) F={original_f} L={work_l} "
                f"subject={'absent' if is_empty else ('secondary-only' if allow_secondary_humans else 'present')} "
                f"picture_refs={_picture_refs_report(inject_target_references, picture1, picture2, picture3)}"
            )
            del (
                depth_shot, first_positive, first_latent, _first_output, first_clean, split,
                upscaled_video, joined, second_positive, _unused_latent, second_output,
                _second_clean, decoded, trimmed,
            )
        if output_images is None:
            raise RuntimeError("H3HardCutRunner C produced no decoded shot")
        return (output_images, float(fps), int(output_images.shape[0]), "\n".join(report_lines))


class H3HardCutRunnerSceneVLMFunControl(H3HardCutRunnerSceneVLM):
    """Root-cause candidate D: move Depth to Fun ControlNet, not ref_video."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        required = dict(spec["required"])
        required.pop("depth")
        required["control_video"] = ("IMAGE",)
        required["control_model_patch"] = ("MODEL_PATCH",)
        required["control_strength"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
        required["control_start_percent"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})
        required["control_end_percent"] = ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001})
        required["control_second_pass"] = ("BOOLEAN", {"default": False})
        return {"required": required, "optional": spec.get("optional", {})}

    def run(
        self,
        model,
        clip,
        video_vae,
        rgb,
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
        control_video,
        control_model_patch,
        control_strength,
        control_start_percent,
        control_end_percent,
        control_second_pass,
        manual_force_empty_indices="",
        picture3=None,
    ):
        if control_video is None or int(rgb.shape[0]) != int(control_video.shape[0]):
            raise ValueError(
                "RGB/Control frame count mismatch: "
                f"{int(rgb.shape[0])} vs {0 if control_video is None else int(control_video.shape[0])}"
            )
        shots = _detect_hard_cuts(rgb, cut_threshold, min_shot_frames)
        scene_prompts = _parse_scene_prompts(shot_prompts)
        empty_shots = _parse_empty_shots(empty_shot_indices, shot_count=len(shots))
        empty_shots |= _parse_manual_force_empty_indices(
            manual_force_empty_indices, shot_count=len(shots)
        )
        _validate_scene_prompts(scene_prompts, len(shots))
        total_frames = sum(int(end - start) for start, end in shots)
        output_images: torch.Tensor | None = None
        output_cursor = 0
        report_lines = [
            "H3HardCutRunner D: shot-local VLM + Fun ControlNet root-cause candidate",
            f"shots={len(shots)}, fps={float(fps):.6g}, threshold={float(cut_threshold):.4g}",
            f"picture3={'present' if picture3 is not None else 'omitted'}, control_second_pass={bool(control_second_pass)}",
            f"manual_force_empty_indices={str(manual_force_empty_indices or '').strip() or 'none'}",
            "depth_transport=control_video; h3_ref_video=omitted",
        ]
        for shot_id, (start, end) in enumerate(shots):
            original_f = int(end - start)
            scene_record = scene_prompts[shot_id + 1]
            policy = _resolve_shot_policy(scene_record, shot_id + 1, empty_shots)
            is_empty = bool(policy["strict_empty"])
            allow_secondary_humans = bool(policy["allow_secondary_humans"])
            inject_target_references = bool(policy["inject_target_references"])
            shot_seed = _derive_shot_seed(seed, shot_id)
            work_l = _legal_length(original_f)
            control_shot = _fit_length(control_video[start:end], work_l)
            control_model = _apply_h3_fun_control(
                model,
                control_model_patch,
                video_vae,
                control_shot,
                control_strength,
                control_start_percent,
                control_end_percent,
            )
            first_positive, first_latent = _h3_ref2v_with_picture3(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(
                    scene_prompts, shot_id, str(prompt_first or ""), is_empty, "first",
                    allow_secondary_humans=allow_secondary_humans,
                    picture3_present=_has_reference_tensor(picture3),
                ),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=picture1 if inject_target_references else None,
                picture2=picture2 if inject_target_references else None,
                picture3=picture3 if inject_target_references else None,
                depth=None,
            )
            _first_output, first_clean = _sample(
                control_model, first_positive, sampler, high_sigmas, first_latent, shot_seed
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
                "LTXVConcatAVLatent", {"video_latent": upscaled_video, "audio_latent": split[1]}
            )[0]
            second_positive, _unused_latent = _h3_ref2v_with_picture3(
                clip=clip,
                vae=video_vae,
                prompt=_compose_shot_prompt(
                    scene_prompts, shot_id, str(prompt_second or ""), is_empty, "second",
                    allow_secondary_humans=allow_secondary_humans,
                    picture3_present=_has_reference_tensor(picture3),
                ),
                width=int(base_width),
                height=int(base_height),
                length=work_l,
                picture1=picture1 if inject_target_references else None,
                picture2=picture2 if inject_target_references else None,
                picture3=picture3 if inject_target_references else None,
                depth=None,
            )
            second_model = control_model if bool(control_second_pass) else model
            second_output, _second_clean = _sample(
                second_model, second_positive, sampler, low_sigmas, joined, shot_seed
            )
            decoded = _call_node("VAEDecode", {"samples": second_output, "vae": video_vae})[0]
            trimmed = decoded[:original_f].detach().to(device="cpu", dtype=torch.float32)
            if output_images is None:
                output_images = torch.empty(
                    (total_frames, *tuple(trimmed.shape[1:])), dtype=trimmed.dtype, device="cpu"
                )
            elif tuple(output_images.shape[1:]) != tuple(trimmed.shape[1:]):
                raise RuntimeError("Shot decode shape changed across hard-cut segments")
            output_images[output_cursor : output_cursor + original_f].copy_(trimmed)
            output_cursor += original_f
            report_lines.append(
                f"shot={shot_id + 1} range=[{start},{end}) F={original_f} L={work_l} "
                f"subject={'absent' if is_empty else ('secondary-only' if allow_secondary_humans else 'present')} "
                f"picture_refs={_picture_refs_report(inject_target_references, picture1, picture2, picture3)}"
            )
            del (
                control_shot, control_model, first_positive, first_latent, _first_output, first_clean,
                split, upscaled_video, joined, second_positive, _unused_latent, second_output,
                second_model, _second_clean, decoded, trimmed,
            )
        if output_images is None:
            raise RuntimeError("H3HardCutRunner D produced no decoded shot")
        return (output_images, float(fps), int(output_images.shape[0]), "\n".join(report_lines))


NODE_CLASS_MAPPINGS = {
    "H3ShotSceneVLM": H3ShotSceneVLM,
    "H3OptionalPicture3": H3OptionalPicture3,
    "H3HardCutRunnerSceneVLM": H3HardCutRunnerSceneVLM,
    "H3HardCutRunnerSceneVLMFunControl": H3HardCutRunnerSceneVLMFunControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ShotSceneVLM": "H3 Shot Scene VLM (Per-Shot, Dynamic)",
    "H3OptionalPicture3": "H3 Optional Picture3 Upload",
    "H3HardCutRunnerSceneVLM": "H3 Hard-Cut Runner C (Scene VLM + Picture3)",
    "H3HardCutRunnerSceneVLMFunControl": "H3 Hard-Cut Runner D (Scene VLM + Fun ControlNet)",
}

