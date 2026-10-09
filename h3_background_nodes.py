"""ComfyUI adapters for the audited task-local background reference core.

This module intentionally contains no matching, caching, or washing algorithm.  It
only converts ComfyUI values, validates node contracts, and delegates to the
background core modules.
"""
import json
import uuid

import numpy as np

if __package__:
    from .h3_background_cache import BackgroundCache
    from .h3_background_contracts import BackgroundView
    from .h3_best_background_frame import select_best_frame
    from .h3_environment_matcher import EnvironmentMatcher
else:
    from h3_background_cache import BackgroundCache
    from h3_background_contracts import BackgroundView
    from h3_best_background_frame import select_best_frame
    from h3_environment_matcher import EnvironmentMatcher


CATEGORY = "video/MiniMaxH3/C Background"


def _numpy(value, name):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    value = np.asarray(value)
    if value.size == 0:
        raise ValueError(f"{name} must not be empty")
    return value


def _frames(value, name="shot_rgb_raw"):
    value = _numpy(value, name)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be [F,H,W,C] RGB")
    return value


def _optional_masks(value):
    if value is None:
        return None
    value = _numpy(value, "person_masks")
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError("person_masks must be [F,H,W] or [F,H,W,1]")
    return value


def _optional_depth(value):
    if value is None:
        return None
    value = _numpy(value, "depth")
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim not in (3, 4):
        raise ValueError("depth must be [F,H,W] or [F,H,W,C]")
    return value


def _one_image(value, name):
    value = _numpy(value, name)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one image")
        value = value[0]
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be [H,W,C] RGB")
    return value


def _one_mask(value):
    value = _numpy(value, "removal_mask")
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError("removal_mask must contain exactly one mask")
        value = value[0]
    if value.ndim != 2:
        raise ValueError("removal_mask must be [H,W] or [1,H,W]")
    return value


def _image_output(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        # Core records are immutable. A downstream IMAGE consumer may write its
        # tensor, so copy only this selected image, not the full Shot batch.
        array = np.array(array, dtype=np.float32, order="C", copy=True)
    import torch
    return torch.from_numpy(np.ascontiguousarray(array))


def _mask_output(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2:
        array = array[None, ...]
    import torch
    return torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))


class BackgroundTaskHandle:
    """Explicit per-video handle; no module-global cache is used."""

    def __init__(self, task_id, db_path=":memory:"):
        self.task_id = str(task_id)
        self.db_path = str(db_path)
        self.cache = BackgroundCache(self.task_id, self.db_path)
        self.matcher = EnvironmentMatcher()
        self.cleaned = False

    def cleanup(self):
        if not self.cleaned:
            self.cache.cleanup()
            self.cleaned = True


class H3CBackgroundTaskStart:
    @classmethod
    def IS_CHANGED(cls, task_id="", db_path=":memory:"):
        # Each execution is a new video, including an unchanged input filename.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task_id": ("STRING", {"default": "", "multiline": False}),
            "db_path": ("STRING", {"default": ":memory:", "multiline": False}),
        }}

    RETURN_TYPES = ("BACKGROUND_TASK", "STRING")
    RETURN_NAMES = ("task", "task_id")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, task_id="", db_path=":memory:"):
        # The widget is a label, never an API to resume another video's cache.
        label = str(task_id).strip() or "h3bg"
        task_id = f"{label}-{uuid.uuid4().hex}"
        db_path = str(db_path).strip() or ":memory:"
        task = BackgroundTaskHandle(task_id, db_path)
        return task, task_id


class H3CBestBackgroundFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": ("BACKGROUND_TASK",),
            "shot_rgb_raw": ("IMAGE",),
            "shot_number": ("INT", {"default": 1, "min": 1}),
            "background_only": ("BOOLEAN", {"default": False}),
            "analysis_edge": ("INT", {"default": 320, "min": 16}),
            "candidate_limit": ("INT", {"default": 5, "min": 1}),
        }, "optional": {
            "person_masks": ("MASK",),
            "depth": ("IMAGE",),
        }}

    RETURN_TYPES = ("BACKGROUND_CANDIDATE", "IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("candidate", "selected_frame", "selected_mask", "selected_index", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, task, shot_rgb_raw, shot_number, background_only, analysis_edge, candidate_limit,
            person_masks=None, depth=None):
        if not isinstance(task, BackgroundTaskHandle) or task.cleaned:
            raise ValueError("task must be an active BACKGROUND_TASK")
        candidate = select_best_frame(
            _frames(shot_rgb_raw), task.task_id, shot_number,
            person_masks=_optional_masks(person_masks), depth=_optional_depth(depth),
            analysis_edge=analysis_edge, candidate_limit=candidate_limit,
            background_only=background_only,
        )
        return (candidate, _image_output(candidate.image), _mask_output(candidate.foreground_mask),
                int(candidate.frame_index), json.dumps(candidate.report, ensure_ascii=False, allow_nan=False))


class H3CBackgroundResolve:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": ("BACKGROUND_TASK",),
            "candidate": ("BACKGROUND_CANDIDATE",),
            "revision": ("STRING", {"default": "default", "multiline": False}),
        }}

    RETURN_TYPES = ("BACKGROUND_VIEW", "BACKGROUND_DECISION", "IMAGE", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("view", "decision", "clean_background", "view_id", "state", "ready")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, task, candidate, revision="default"):
        if not isinstance(task, BackgroundTaskHandle) or task.cleaned:
            raise ValueError("task must be an active BACKGROUND_TASK")
        if getattr(candidate, "task_id", None) != task.task_id:
            raise ValueError("candidate belongs to a different background task")
        decision, view = task.cache.resolve(candidate, task.matcher, str(revision) or "default")
        clean = None if view is None or view.state != "READY" else view.clean_background
        view_id = "" if view is None else view.view_id
        state = "" if view is None else view.state
        return (view, decision, _image_output(clean), view_id, state, bool(clean is not None))


class H3CBackgroundWashClaim:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"task": ("BACKGROUND_TASK",), "view": ("BACKGROUND_VIEW",)}}

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("wash_token", "claimed", "view_id")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, task, view):
        if not isinstance(task, BackgroundTaskHandle) or task.cleaned:
            raise ValueError("task must be an active BACKGROUND_TASK")
        if not isinstance(view, BackgroundView) or view.task_id != task.task_id:
            raise ValueError("view belongs to a different background task")
        token = task.cache.claim_wash(view.view_id)
        return (token or "", bool(token), view.view_id)


class H3CBackgroundAttachClean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": ("BACKGROUND_TASK",),
            "view": ("BACKGROUND_VIEW",),
            "clean_background": ("IMAGE",),
            "wash_token": ("STRING", {"forceInput": True}),
        }, "optional": {
            "removal_mask": ("MASK",),
        }}

    RETURN_TYPES = ("BACKGROUND_VIEW", "IMAGE", "STRING")
    RETURN_NAMES = ("view", "clean_background", "report")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, task, view, clean_background, removal_mask=None, wash_token=""):
        if not isinstance(task, BackgroundTaskHandle) or task.cleaned:
            raise ValueError("task must be an active BACKGROUND_TASK")
        if not isinstance(view, BackgroundView) or view.task_id != task.task_id:
            raise ValueError("view belongs to a different background task")
        if not isinstance(wash_token, str) or not wash_token.strip():
            raise ValueError("automatic attach requires a nonempty claim_wash token")
        updated = task.cache.attach_clean(
            view.view_id, _one_image(clean_background, "clean_background"),
            removal_mask=None if removal_mask is None else _one_mask(removal_mask),
            wash_token=wash_token,
        )
        if updated.state != "READY" or updated.clean_background is None:
            raise ValueError("attach_clean did not produce a READY background")
        return updated, _image_output(updated.clean_background), f"view={updated.view_id} state={updated.state}"


class H3CBackgroundTaskCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "task": ("BACKGROUND_TASK",),
            "completion": ("*", {"forceInput": True}),
        }}

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("report", "cleaned")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, task, completion):
        # Completion is a graph dependency on ALL final consumers, not a value
        # to interpret. None is valid for a host output with no payload.
        if not isinstance(task, BackgroundTaskHandle):
            raise ValueError("task must be a BACKGROUND_TASK")
        task.cleanup()
        return f"task={task.task_id} cleaned=true", True


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    H3CBackgroundTaskStart,
    H3CBestBackgroundFrame,
    H3CBackgroundResolve,
    H3CBackgroundWashClaim,
    H3CBackgroundAttachClean,
    H3CBackgroundTaskCleanup,
)}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CBackgroundTaskStart": "H3 C Background Task Start",
    "H3CBestBackgroundFrame": "H3 C Best Background Frame",
    "H3CBackgroundResolve": "H3 C Background Match / Resolve",
    "H3CBackgroundWashClaim": "H3 C Background Wash Claim",
    "H3CBackgroundAttachClean": "H3 C Background Attach Clean",
    "H3CBackgroundTaskCleanup": "H3 C Background Task Cleanup",
}
