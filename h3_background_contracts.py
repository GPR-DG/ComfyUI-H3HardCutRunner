"""Shared CPU contracts. No imports from ComfyUI or the production H3 plugin."""
from dataclasses import dataclass, field
import hashlib
import json
import uuid
from numbers import Real
import numpy as np


def require_id(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f'{name} must be a non-empty string of <=200 characters')


def require_rgb(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('RGB must be a non-empty numpy HWC array with 3 channels')
    if image.dtype == np.uint8:
        return image
    if image.dtype.kind != 'f' or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('RGB must be uint8 [0,255] or finite float [0,1]; no implicit alpha/BGR conversion')
    return image


def rgb8(image):
    require_rgb(image)
    return image if image.dtype == np.uint8 else np.rint(image * 255).astype(np.uint8)


def require_mask(mask, shape):
    if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype.kind not in 'bufi':
        raise ValueError('foreground mask must match H/W (foreground=1)')
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError('foreground mask must be finite [0,1]')
    return mask


def json_safe(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ValueError('metadata/report must be finite JSON data') from e


def score_scalar(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError(f'{name} must be a real scalar, not bool/array')
    value=float(value)
    if not np.isfinite(value) or not 0<=value<=1:
        raise ValueError(f'{name} must be finite [0,1]')
    return value


@dataclass(frozen=True)
class CandidateBackgroundRecord:
    task_id: str
    shot_number: int
    frame_index: int
    image: np.ndarray
    quality_score: float
    foreground_mask: np.ndarray | None = None
    depth: np.ndarray | None = None
    background_only: bool = False
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        require_id(self.task_id, 'task_id')
        for name, minimum in [('shot_number', 1), ('frame_index', 0)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        require_rgb(self.image)
        if not isinstance(self.background_only, bool):
            raise ValueError('background_only must be an explicit bool')
        object.__setattr__(self,'quality_score',score_scalar(self.quality_score,'quality_score'))
        if self.foreground_mask is not None:
            require_mask(self.foreground_mask, self.image.shape[:2])
        if self.depth is not None:
            if (not isinstance(self.depth, np.ndarray) or self.depth.ndim not in (2,3)
                    or self.depth.shape[:2] != self.image.shape[:2] or self.depth.size==0 or self.depth.dtype.kind not in 'ufi'
                    or not np.isfinite(self.depth).all()):
                raise ValueError('depth must have aligned H/W and finite numeric pixels')
        # Own ONE selected frame, not a view keeping a whole Shot tensor/storage alive.
        for name in ('image','foreground_mask','depth'):
            array = getattr(self,name)
            if array is not None:
                array = array.copy(); array.setflags(write=False)
                object.__setattr__(self,name,array)
        object.__setattr__(self,'report',json_safe(self.report))

    @property
    def key(self):
        h = hashlib.sha256(json.dumps([self.task_id,self.shot_number,self.frame_index,
                                      self.quality_score,self.background_only]).encode())
        for array in (self.image,self.foreground_mask,self.depth):
            if array is not None:
                h.update(str((array.shape,array.dtype.str)).encode());h.update(array.tobytes())
            else:
                h.update(b'None')
        return h.hexdigest()


@dataclass(frozen=True)
class EnvironmentDecision:
    task_id: str
    candidate_key: str
    decision: str
    matched_environment_id: str | None = None
    matched_view_id: str | None = None
    global_similarity: float | None = None
    geometry_score: float | None = None
    semantic_score: float | None = None
    confidence: float = 0.0
    reason: str = ''
    report: dict = field(default_factory=dict)
    # A fresh manual decision means a new revision; retries must keep this key.
    # The production matcher supplies a deterministic key for reconstructed retries.
    decision_key: str = field(default_factory=lambda:uuid.uuid4().hex)

    def __post_init__(self):
        require_id(self.task_id,'task_id'); require_id(self.candidate_key,'candidate_key')
        require_id(self.decision_key,'decision_key')
        if self.decision not in {'REUSE_EXISTING','NEW_VIEW','NEW_ENVIRONMENT','UNCERTAIN'}:
            raise ValueError('unknown environment decision')
        expected = {'REUSE_EXISTING':(True,True),'NEW_VIEW':(True,False),
                    'NEW_ENVIRONMENT':(False,False),'UNCERTAIN':(False,False)}[self.decision]
        if tuple(v is not None for v in (self.matched_environment_id,self.matched_view_id)) != expected:
            raise ValueError('decision IDs do not match decision kind')
        for name in ('matched_environment_id','matched_view_id'):
            if getattr(self,name) is not None: require_id(getattr(self,name),name)
        for name in ('global_similarity','geometry_score','semantic_score','confidence'):
            value=getattr(self,name)
            if value is not None or name=='confidence':
                object.__setattr__(self,name,score_scalar(value,name))
        object.__setattr__(self,'report',json_safe(self.report))


@dataclass
class BackgroundView:
    task_id: str
    environment_id: str
    view_id: str
    state: str
    candidate: CandidateBackgroundRecord
    features: dict = field(default_factory=dict)
    clean_background: np.ndarray | None = None
    clean_loaded: bool = True  # index snapshot omits clean BLOB; get/apply load it.
    # Auxiliary original-frame removal support (or conservative legacy diff),
    # never replaces raw evidence or provides synthetic clean-image geometry.
    reuse_exclusion_mask: np.ndarray | None = None
    reuse_features: dict = field(default_factory=dict)  # separately keyed auxiliary evidence
