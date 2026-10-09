"""Two-stage task-local storage, not an image matcher. SQLite owns transactions."""
import io
import json
import sqlite3
import uuid
import tempfile
import hashlib
from pathlib import Path
from contextlib import contextmanager
from dataclasses import asdict,replace
import numpy as np
try:
    from .h3_background_contracts import (BackgroundView, CandidateBackgroundRecord, EnvironmentDecision,
                                         require_id, require_rgb, require_mask, rgb8)
except ImportError:
    from h3_background_contracts import (BackgroundView, CandidateBackgroundRecord, EnvironmentDecision,
                                         require_id, require_rgb, require_mask, rgb8)


def _pack(arrays):
    for key,value in arrays.items():
        if not isinstance(key,str) or not isinstance(value,np.ndarray) or value.dtype.kind not in 'bufiUS':
            raise ValueError('stored features/images must be numeric/string numpy arrays, never pickle objects')
        if value.dtype.kind in 'ufi' and not np.isfinite(value).all():
            raise ValueError('stored arrays must be finite')
    buffer=io.BytesIO(); np.savez_compressed(buffer,**arrays)
    return buffer.getvalue()


def _unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}


class BackgroundCache:
    """Use a NEW task_id per video. Reusing the same ID explicitly resumes that task.

    Connection is single-threaded (SQLite default), no hidden global state/eviction.
    ':memory:' is task-local; pass a SQLite path to opt into lossless persistence.
    """
    def __init__(self,task_id,db_path=':memory:'):
        require_id(task_id,'task_id');self.task_id=task_id
        self._db_path=':memory:' if str(db_path)==':memory:' else str(Path(db_path).resolve())
        self._cleaned=False
        self.db=sqlite3.connect(self._db_path,timeout=30)
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        version=self.db.execute('PRAGMA user_version').fetchone()[0]
        if version not in (0,1,2,3):
            self.db.close();raise ValueError('unsupported background cache schema version')
        self.db.executescript('''
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS environments (
                task_id TEXT NOT NULL, env_id TEXT NOT NULL, PRIMARY KEY(task_id,env_id));
            CREATE TABLE IF NOT EXISTS views (
                task_id TEXT NOT NULL, view_id TEXT NOT NULL, env_id TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('PENDING','READY')),
                metadata TEXT NOT NULL, candidate BLOB NOT NULL, features BLOB NOT NULL,
                clean BLOB, PRIMARY KEY(task_id,view_id),
                FOREIGN KEY(task_id,env_id) REFERENCES environments(task_id,env_id),
                CHECK((state='PENDING' AND clean IS NULL) OR (state='READY' AND clean IS NOT NULL)));
            CREATE TABLE IF NOT EXISTS decisions (
                task_id TEXT NOT NULL, decision_key TEXT NOT NULL, binding TEXT NOT NULL,
                view_id TEXT NOT NULL, decision TEXT, PRIMARY KEY(task_id,decision_key),
                FOREIGN KEY(task_id,view_id) REFERENCES views(task_id,view_id));
            CREATE TABLE IF NOT EXISTS ended_tasks (task_id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS wash_jobs (
                task_id TEXT NOT NULL, view_id TEXT NOT NULL, token TEXT NOT NULL,
                result_digest TEXT, PRIMARY KEY(task_id,view_id),
                FOREIGN KEY(task_id,view_id) REFERENCES views(task_id,view_id));
        ''')
        with self.db:
            if 'decision' not in {r['name'] for r in self.db.execute('PRAGMA table_info(decisions)')}:
                self.db.execute('ALTER TABLE decisions ADD COLUMN decision TEXT')
            self.db.execute('PRAGMA user_version=3')
        try:self._require_active()
        except Exception:self.close();raise

    def _require_active(self):
        if self.db is None:raise ValueError('background cache connection is closed')
        if self.db.execute('SELECT 1 FROM ended_tasks WHERE task_id=?',(self.task_id,)).fetchone():
            raise ValueError('background task has ended; start a fresh task scope')

    def close(self):
        if self.db is not None:self.db.close();self.db=None

    def cleanup(self):
        """After ALL background consumers/workers finish, delete only this task.

        close() intentionally supports explicit same-task recovery; cleanup() ends it.
        Shared DB files are not unlinked, because other tasks may still own rows.
        """
        if self._cleaned:return
        if self.db is None and self._db_path==':memory:':self._cleaned=True;return
        conn=self.db if self.db is not None else sqlite3.connect(self._db_path,timeout=30)
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('INSERT OR IGNORE INTO ended_tasks VALUES (?)',(self.task_id,))
                for table in ('wash_jobs','decisions','views','environments'):
                    conn.execute(f'DELETE FROM {table} WHERE task_id=?',(self.task_id,))
            self._cleaned=True
        finally:
            conn.close();self.db=None

    @classmethod
    @contextmanager
    def task_scope(cls,temp_root=None):
        """Fresh per-video ID + owned temporary DB; clean on success/failure/cancel.

        The caller owns this context across the WHOLE video, never one context per Shot.
        Process kill cannot execute finally, but the next task always gets a fresh scope.
        """
        with tempfile.TemporaryDirectory(prefix='h3-background-',dir=temp_root) as directory:
            cache=cls(uuid.uuid4().hex,Path(directory)/'background_cache.sqlite3')
            try:yield cache
            finally:cache.cleanup()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def _view(self,row,clean_loaded=True):
        if row is None: raise ValueError('view not found in current task')
        arrays=_unpack(row['candidate']); m=json.loads(row['metadata'])
        rec=CandidateBackgroundRecord(self.task_id,m['shot_number'],m['frame_index'],arrays['image'],
                                      m['quality_score'],arrays.get('mask'),arrays.get('depth'),
                                      m['background_only'],m['report'])
        clean=None if row['clean'] is None else _unpack(row['clean'])['image']
        return BackgroundView(self.task_id,row['env_id'],row['view_id'],row['state'],rec,
                              _unpack(row['features']),clean,clean_loaded,arrays.get('reuse_exclusion'),
                              {k.removeprefix('reuse_feature_'):v for k,v in arrays.items() if k.startswith('reuse_feature_')})

    def get(self,view_id):
        self._require_active()
        require_id(view_id,'view_id')
        return self._view(self.db.execute('SELECT * FROM views WHERE task_id=? AND view_id=?',
                                         (self.task_id,view_id)).fetchone())

    def index(self):
        self._require_active()
        # Full history, but never unpack all high-resolution clean backgrounds.
        # Candidate images remain available for invalid/stale-feature rebuilding.
        return [self._view(row,clean_loaded=False) for row in self.db.execute(
            'SELECT task_id,view_id,env_id,state,metadata,candidate,features,NULL AS clean '
            'FROM views WHERE task_id=? ORDER BY rowid',(self.task_id,))]

    def apply(self,candidate,decision,features=None):
        self._require_active()
        if candidate.task_id != self.task_id or decision.task_id != self.task_id or decision.candidate_key != candidate.key:
            raise ValueError('stale decision or cross-task candidate/decision')
        if self.db.in_transaction:return self._apply(candidate,decision,features)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            return self._apply(candidate,decision,features)

    def _apply(self,candidate,decision,features):
        self._require_active()  # also inside transaction, after acquiring its write lock
        kind=decision.decision
        binding=json.dumps([candidate.key,kind,decision.matched_environment_id,decision.matched_view_id])
        existing=self.db.execute('SELECT binding,view_id FROM decisions WHERE task_id=? AND decision_key=?',
                                 (self.task_id,decision.decision_key)).fetchone()
        if existing:
            if existing['binding']!=binding:
                raise ValueError('decision_key reused for different candidate/action/target; use a new revision key')
            return self.get(existing['view_id'])
        if kind=='UNCERTAIN': return None  # No guessed IDs, no cache mutation or fake clean image.
        if kind=='REUSE_EXISTING':
            view=self.get(decision.matched_view_id)
            if view.environment_id != decision.matched_environment_id:
                raise ValueError('environment/view mismatch')
            self._receipt(decision,binding,view.view_id)
            return view  # PENDING is truthful: clean_background=None, not representative image.
        arrays={'image':candidate.image}
        if candidate.foreground_mask is not None: arrays['mask']=candidate.foreground_mask
        if candidate.depth is not None: arrays['depth']=candidate.depth
        packed=_pack(arrays);packed_features=_pack(features if features is not None else {})
        meta=json.dumps(dict(shot_number=candidate.shot_number,frame_index=candidate.frame_index,
                             quality_score=candidate.quality_score,background_only=candidate.background_only,
                             report=candidate.report),allow_nan=False)
        env=decision.matched_environment_id or uuid.uuid4().hex; view_id=uuid.uuid4().hex
        if kind=='NEW_ENVIRONMENT':
            self.db.execute('INSERT INTO environments VALUES (?,?)',(self.task_id,env))
        elif not self.db.execute('SELECT 1 FROM environments WHERE task_id=? AND env_id=?',(self.task_id,env)).fetchone():
            raise ValueError('environment not found in current task')
        self.db.execute('INSERT INTO views VALUES (?,?,?,?,?,?,?,NULL)',
                        (self.task_id,view_id,env,'PENDING',meta,packed,packed_features))
        self._receipt(decision,binding,view_id)
        return self.get(view_id)

    def _receipt(self,decision,binding,view_id):
        self.db.execute('INSERT INTO decisions (task_id,decision_key,binding,view_id,decision) VALUES (?,?,?,?,?)',
                        (self.task_id,decision.decision_key,binding,view_id,json.dumps(asdict(decision),allow_nan=False)))

    def resolve(self,candidate,matcher,revision='default'):
        """Production transaction: receipt -> full history match -> apply.

        One short SQLite write lock covers the decision, not external washing.
        Call this instead of a non-atomic index/match/apply sequence in wrappers.
        """
        try:
            from .h3_environment_matcher import production_request_key,extract_features
        except ImportError:
            from h3_environment_matcher import production_request_key,extract_features
        self._require_active()
        if candidate.task_id!=self.task_id:raise ValueError('cross-task candidate')
        key=production_request_key(candidate,revision)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            self._require_active()
            row=self.db.execute('SELECT * FROM decisions WHERE task_id=? AND decision_key=?',
                                (self.task_id,key)).fetchone()
            if row:
                binding=json.loads(row['binding'])
                if binding[0]!=candidate.key:raise ValueError('request/candidate mismatch')
                d=(EnvironmentDecision(**json.loads(row['decision'])) if row['decision'] else
                   EnvironmentDecision(self.task_id,candidate.key,*binding[1:],decision_key=key,
                                       reason='restored persisted decision binding'))
                view=self.get(row['view_id'])
                return replace(d,report={**d.report,'receipt_replayed':True,
                               'production_action':'USE_READY_REFERENCE' if view.state=='READY' else 'ENSURE_WASH_EXISTING_VIEW'}),view
            views=self.index()
            refreshed=self._refresh_reuse_features(views)
            d=matcher.match_for_production(candidate,views,revision=revision)
            d=replace(d,report={**d.report,'cached_reuse_features_refreshed':refreshed})
            if d.decision_key!=key:raise ValueError('matcher must preserve production request key')
            features=extract_features(candidate) if d.decision in ('NEW_ENVIRONMENT','NEW_VIEW') else None
            return d,self.apply(candidate,d,features)

    def _refresh_reuse_features(self,views):
        # Called inside resolve's write transaction; old/version-stale evidence
        # is rebuilt once and persists. Standalone matcher remains side-effect free.
        try:
            from .h3_environment_matcher import _matching_record,_valid_features,extract_features
        except ImportError:
            from h3_environment_matcher import _matching_record,_valid_features,extract_features
        refreshed=0
        for view in views:
            record=_matching_record(view)
            if record is view.candidate or _valid_features(view.reuse_features,record):continue
            view.reuse_features=extract_features(record)
            row=self.db.execute('SELECT candidate FROM views WHERE task_id=? AND view_id=?',
                                (self.task_id,view.view_id)).fetchone()
            arrays=_unpack(row[0])
            arrays={k:v for k,v in arrays.items() if not k.startswith('reuse_feature_')}
            arrays.update({'reuse_feature_'+k:v for k,v in view.reuse_features.items()})
            self.db.execute('UPDATE views SET candidate=? WHERE task_id=? AND view_id=?',
                            (_pack(arrays),self.task_id,view.view_id))
            refreshed+=1
        return refreshed

    def claim_wash(self,view_id):
        """Only the winner may dispatch an external wash; no lease/auto retry."""
        require_id(view_id,'view_id');self._require_active()
        with self.db:
            self.db.execute('BEGIN IMMEDIATE');self._require_active()
            row=self.db.execute('SELECT state FROM views WHERE task_id=? AND view_id=?',
                                (self.task_id,view_id)).fetchone()
            if row is None:raise ValueError('view not found in current task')
            if row['state']=='READY':return None
            token=uuid.uuid4().hex
            inserted=self.db.execute('INSERT OR IGNORE INTO wash_jobs VALUES (?,?,?,NULL)',
                                     (self.task_id,view_id,token)).rowcount
            return token if inserted else None

    def release_wash(self,view_id,token):
        """Explicit failed/cancelled-job release. Cancel/join worker first."""
        require_id(view_id,'view_id');require_id(token,'wash_token');self._require_active()
        with self.db:
            self.db.execute('BEGIN IMMEDIATE');self._require_active()
            removed=self.db.execute('DELETE FROM wash_jobs WHERE task_id=? AND view_id=? AND token=? AND result_digest IS NULL',
                                    (self.task_id,view_id,token)).rowcount
            if removed!=1:raise ValueError('stale or completed wash token')

    def attach_clean(self,view_id,clean_background,*,removal_mask=None,wash_token=None):
        """Worker: claim token + actual original-frame removal mask. No-token
        calls preserve the explicit manual legacy API, not worker completion.
        """
        view=self.get(view_id);require_rgb(clean_background)
        if clean_background.shape != view.candidate.image.shape:
            raise ValueError('clean background must preserve representative H/W/C; use a new View for new geometry')
        # One owned snapshot binds the stored clean and its auxiliary exclusion.
        # Upstream workers may reuse their buffer after handing it to this boundary.
        clean_background=clean_background.copy()
        if removal_mask is not None:
            require_mask(removal_mask,view.candidate.image.shape[:2]);removal_mask=removal_mask.copy()
        if wash_token is not None:require_id(wash_token,'wash_token')
        blob=_pack({'image':clean_background})
        arrays={'image':view.candidate.image}
        for name,key in [('foreground_mask','mask'),('depth','depth')]:
            if getattr(view.candidate,name) is not None:arrays[key]=getattr(view.candidate,name)
        if view.candidate.foreground_mask is None and not view.candidate.background_only:
            # Trust the upstream clean declaration only for exclusion, not room identity.
            # Changed/generated pixels cannot supply geometry. Match untouched ORIGINAL
            # RGB support against later masked Shots using the same geometry/dense gates.
            # Production washer supplies ORIGINAL-frame foreground/removal support.
            # Pixel difference alone cannot identify a human vs global color editing.
            # Keep exact-diff fallback conservative for legacy callers; never lower
            # a tolerance to "prove" unknown background pixels are foreground-free.
            arrays['reuse_exclusion']=(removal_mask if removal_mask is not None else
                np.any(rgb8(view.candidate.image)!=rgb8(clean_background),axis=-1).astype(np.uint8))
            try:
                from .h3_environment_matcher import extract_features
            except ImportError:
                from h3_environment_matcher import extract_features
            features=extract_features(replace(view.candidate,foreground_mask=arrays['reuse_exclusion']))
            arrays.update({'reuse_feature_'+k:v for k,v in features.items()})
        digest=hashlib.sha256()
        for array in (clean_background,removal_mask):
            if array is None:digest.update(b'None')
            else:
                digest.update(str((array.shape,array.dtype.str)).encode());digest.update(array.tobytes())
        result_digest=digest.hexdigest()
        packed_candidate=_pack(arrays)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            self._require_active()
            job=self.db.execute('SELECT * FROM wash_jobs WHERE task_id=? AND view_id=?',
                                (self.task_id,view_id)).fetchone()
            if wash_token is not None:
                if job is None or job['token']!=wash_token:raise ValueError('stale wash token')
                if job['result_digest'] is not None:
                    if job['result_digest']==result_digest:return self.get(view_id)
                    raise ValueError('completed wash token cannot replace its result')
                if self.get(view_id).state!='PENDING':raise ValueError('wash result requires PENDING view')
            elif job is not None:
                if job['result_digest'] is None:raise ValueError('claimed worker must supply its wash token')
                # Deliberate manual correction invalidates prior worker receipt.
                self.db.execute('DELETE FROM wash_jobs WHERE task_id=? AND view_id=?',(self.task_id,view_id))
            self.db.execute("UPDATE views SET state='READY', clean=?, candidate=? WHERE task_id=? AND view_id=?",
                            (blob,packed_candidate,self.task_id,view_id))
            if wash_token is not None:
                self.db.execute('UPDATE wash_jobs SET result_digest=? WHERE task_id=? AND view_id=? AND token=?',
                                (result_digest,self.task_id,view_id,wash_token))
        return self.get(view_id)



