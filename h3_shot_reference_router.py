"""Shot-local target reference routing; no cropping, inference or tensor copies.

Future canvas wiring:
SceneVLM.shot_prompts + DualRefPad.shot_number + Policy.inject_target_references
-> this node -> Ref2VA.ref_image_0..7. Append reference_contract to the target
prompt AFTER optional-Picture3 cleanup; replace legacy hardcoded Picture roles
in that new canvas. Old workflows/PromptCompiler are deliberately unchanged.
Slot ref_image_8 remains free for a future background reference, not implemented.
"""
from __future__ import annotations

import json

from .h3_hardcut_runner_a import _validate_scene_prompts
from .h3_hardcut_runner_c import _has_reference_tensor, _parse_scene_prompts
from .h3_hardcut_ref_adapters import _require_image_batch


class H3CShotReferenceRouter:
    """Gate first with ShotPolicy, then select/compact one or two person packages.

    Each detail is externally cropped from its named full image; this node
    preserves those associations by input name, but cannot verify pixel identity.
    Optional back_full=None/empty disables the whole Back package. A complete
    Front package is the fallback for every orientation, including back.
    IMAGE outputs contain at most one input image each, never an ImageBatch.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "shot_prompts": ("STRING", {"forceInput": True}),
            "shot_number": ("INT", {"forceInput": True, "min": 1}),
            "inject_target_references": ("BOOLEAN", {"forceInput": True}),
            "front_full": ("IMAGE",),
        }, "optional": {
            "front_detail_1": ("IMAGE",), "front_detail_2": ("IMAGE",),
            "front_detail_3": ("IMAGE",), "back_full": ("IMAGE",),
            "back_detail_1": ("IMAGE",), "back_detail_2": ("IMAGE",),
            "back_detail_3": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE",) * 8 + ("STRING",) * 4
    RETURN_NAMES = tuple(f"ref_image_{i}" for i in range(8)) + (
        "reference_mode", "orientation", "reference_contract", "report",
    )
    FUNCTION = "run"
    CATEGORY = "video/MiniMaxH3/C Modular"

    def run(self, shot_prompts, shot_number, inject_target_references, front_full,
            front_detail_1=None, front_detail_2=None, front_detail_3=None,
            back_full=None, back_detail_1=None, back_detail_2=None, back_detail_3=None):
        if isinstance(shot_number, bool) or not isinstance(shot_number, int) or shot_number < 1:
            raise ValueError("shot_number must be a positive integer")
        if not isinstance(inject_target_references, bool):
            raise ValueError("inject_target_references must be the ShotPolicy BOOLEAN output")
        records = _parse_scene_prompts(shot_prompts)
        _validate_scene_prompts(records, len(records))
        if shot_number not in records:
            raise ValueError(f"shot {shot_number} is missing from scene records")
        record = records[shot_number]
        orientation = record["orientation"]
        has_back = _has_reference_tensor(back_full)
        selected = []
        mode = "none"

        # Policy is authoritative, including manual force-present overrides.
        # No image validation/injection when this shot is empty or secondary-only.
        if inject_target_references:
            _require_image_batch(front_full, "front_full")
            if int(front_full.shape[0]) != 1:
                raise ValueError("front_full must contain exactly one reference image")
            mode = ("front" if orientation == "front" else "front_fallback") if not has_back else (
                "front" if orientation == "front" else "back" if orientation == "back" else "both"
            )
            packages = (
                ("front", (front_full, front_detail_1, front_detail_2, front_detail_3)),
                ("back", (back_full, back_detail_1, back_detail_2, back_detail_3)),
            )
            for package, images in packages:
                if mode != "both" and not mode.startswith(package):
                    continue
                for index, image in enumerate(images):
                    if not _has_reference_tensor(image):
                        continue
                    role = "full" if index == 0 else f"detail_{index}"
                    name = f"{package}_{role}"
                    _require_image_batch(image, name)
                    if int(image.shape[0]) != 1:
                        raise ValueError(f"{name} must contain exactly one reference image")
                    selected.append((name, package, role, image))

        references = [{
            "slot": f"ref_image_{i}", "picture": i + 1, "input": name,
            "package": package, "role": role,
        } for i, (name, package, role, _) in enumerate(selected)]
        contract = ""
        if references:
            descriptions = [
                f"<Picture {row['picture']}> is the {row['package']} {row['role']} target-person reference"
                for row in references
            ]
            same_person = ("Front and Back packages depict the same target person, not two different people. "
                           if mode == "both" else "Selected references depict the same target person. ")
            contract = ("Target-person reference routing: " + "; ".join(descriptions) + ". " + same_person +
                        "Each detail belongs to its named full-image package. References define "
                        "target appearance, not the scene, camera or additional people.")
        report = json.dumps({
            "shot": shot_number, "orientation": orientation,
            "orientation_reason": record["orientation_reason"],
            "orientation_evidence": record["orientation_evidence"],
            "inject_target_references": inject_target_references,
            "back_available": has_back, "reference_mode": mode,
            "references": references, "reserved_background_slot": "ref_image_8",
        }, ensure_ascii=False, separators=(",", ":"))
        images = [item[3] for item in selected]
        return (*images, *([None] * (8-len(images))), mode, orientation, contract, report)


NODE_CLASS_MAPPINGS = {"H3CShotReferenceRouter": H3CShotReferenceRouter}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CShotReferenceRouter": "H3 C Shot Orientation / Reference Package Router",
}
