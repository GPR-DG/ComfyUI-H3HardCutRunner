"""CPU background matching via installed OpenCV, not a room semantic classifier.

hloc's retrieval->local-verification architecture informs ordering, no copied code.
All history is checked for this small-index phase: HS retrieval cannot exclude an
old environment just because lighting/crop changes. No paid/neural model calls.
"""
import cv2
import numpy as np
from dataclasses import replace
import hashlib,json
try:
    from .h3_background_contracts import EnvironmentDecision, rgb8
except ImportError:
    from h3_background_contracts import EnvironmentDecision, rgb8

FEATURE_VERSION='opencv-sift-background-v3/'+cv2.__version__
# Provisional acceptance rules, NOT business-calibrated probability thresholds.
GEOMETRY_LIMITS=dict(min_inliers=18,lowe_ratio=.75,ransac_error=.006,min_inlier_ratio=.65,
                     min_hull=.15,min_cells=8,max_corner_move=.18,min_overlap=.75,max_aspect_change=.1)
DENSE_LIMITS=dict(min_cells=8,min_cell_valid_fraction=.6,min_std=2.,min_ncc=.8,min_agree_fraction=.75,
                  max_color_error=15.,min_pixel_agree_fraction=.9)


def production_request_key(candidate,revision='default'):
    # Request identity is independent of index, outcome and algorithm version.
    if not isinstance(revision,str) or not revision.strip():raise ValueError('revision must be non-empty')
    return hashlib.sha256(json.dumps(['background-request-v1',candidate.task_id,candidate.key,revision]).encode()).hexdigest()


def extract_features(record,max_edge=640):
    if isinstance(max_edge,bool) or not isinstance(max_edge,int) or max_edge<16:
        raise ValueError('max_edge must be an integer >=16')
    image=rgb8(record.image);h,w=image.shape[:2]
    size=(max(1,round(w*min(1,max_edge/max(h,w)))),max(1,round(h*min(1,max_edge/max(h,w)))))
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    raw_valid=np.ones((h,w),np.float32) if record.foreground_mask is None else (record.foreground_mask<.5).astype(np.float32)
    valid=(cv2.resize(raw_valid,size,interpolation=cv2.INTER_AREA)>=1-1e-6).astype(np.uint8)
    # SIFT's mask only gates keypoint CENTERS, not descriptor support; reject support touching foreground below.
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    histogram=cv2.calcHist([cv2.cvtColor(image,cv2.COLOR_RGB2HSV)],[0,1],valid,[24,16],[0,180,0,256])
    histogram/=max(float(histogram.sum()),1.)
    points=[];descriptors=[]
    if min(gray.shape)>=16 and valid.any():
        keypoints,desc=cv2.SIFT_create(nfeatures=1200).detectAndCompute(gray,valid*255)
        if desc is not None:
            invalid_integral=cv2.integral((valid==0).astype(np.uint8))
            for kp,d in zip(keypoints,desc):
                x,y=kp.pt; radius=max(2,int(np.ceil(kp.size*3)))
                x0,y0=max(0,int(x)-radius),max(0,int(y)-radius)
                x1,y1=min(size[0],int(x)+radius+1),min(size[1],int(y)+radius+1)
                blocked=invalid_integral[y1,x1]-invalid_integral[y0,x1]-invalid_integral[y1,x0]+invalid_integral[y0,x0]
                if blocked==0:
                    points.append((x/size[0],y/size[1]));descriptors.append(d)
    return dict(signature=np.array(FEATURE_VERSION),source_key=np.array(record.key),
                max_edge=np.array(max_edge),histogram=histogram.astype(np.float32),
                points=np.array(points,np.float32).reshape(-1,2),
                descriptors=np.array(descriptors,np.float32).reshape(-1,128),
                visible=np.array(float(raw_valid.mean())),aspect=np.array(w/h))


def _valid_features(features,record):
    """Cache is an optimization, not a trust boundary: stale/invalid data is rebuilt."""
    if not isinstance(features,dict):return False
    for key,expected in [('signature',FEATURE_VERSION),('source_key',record.key),('max_edge',640)]:
        value=features.get(key)
        if not isinstance(value,np.ndarray) or value.shape!=() or value.item()!=expected:return False
    hist,points,desc=[features.get(k) for k in ('histogram','points','descriptors')]
    if any(not isinstance(v,np.ndarray) or v.dtype!=np.float32 or not np.isfinite(v).all()
           for v in (hist,points,desc)):return False
    if (hist.shape!=(24,16) or points.ndim!=2 or points.shape[1]!=2
            or desc.shape!=(len(points),128) or (points<0).any() or (points>1).any()
            or (desc<0).any() or (desc>255).any()
            or (hist<0).any()):return False
    # Fully excluded background is valid NEGATIVE evidence, not stale data.
    # Keep its empty feature cache; geometry still cannot identify/reuse it.
    if not (np.isclose(hist.sum(),1,atol=1e-4) or (not hist.any() and len(points)==0)):return False
    # This signature is raw OpenCV SIFT: normalized to 512 then rounded/clipped.
    # A generous upper bound tolerates rounding; all-zero rows carry no feature.
    norms=np.linalg.norm(desc,axis=1)
    if (norms==0).any() or (norms>600).any():return False
    for key in ('visible','aspect'):
        value=features.get(key)
        if (not isinstance(value,np.ndarray) or value.shape!=() or value.dtype.kind not in 'ufi'
                or not np.isfinite(value)):return False
    return 0<=float(features['visible'])<=1 and np.isclose(float(features['aspect']),record.image.shape[1]/record.image.shape[0])


def _geometry(a,b):
    result=dict(inliers=0,inlier_ratio=0.,coverage_query=0.,coverage_view=0.,occupied_query_cells=0,occupied_view_cells=0,
                overlap_query=0.,overlap_view=0.,corner_displacement=None,identity=False,reusable=False,score=0.)
    da,db=a['descriptors'],b['descriptors']
    limits=GEOMETRY_LIMITS
    if min(len(da),len(db))<limits['min_inliers']:return result
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def accepted(pairs):
        return {m[0].queryIdx:m[0].trainIdx for m in pairs if len(m)==2 and m[0].distance<limits['lowe_ratio']*m[1].distance}
    forward=accepted(matcher.knnMatch(da,db,k=2));reverse=accepted(matcher.knnMatch(db,da,k=2))
    pairs=[(i,j) for i,j in forward.items() if reverse.get(j)==i]
    if len(pairs)<limits['min_inliers']:return result
    qa=a['points'][[i for i,j in pairs]];qb=b['points'][[j for i,j in pairs]]
    H,mask=cv2.findHomography(qa,qb,cv2.RANSAC,limits['ransac_error'],maxIters=3000,confidence=.995)
    if H is None or mask is None or not np.isfinite(H).all():return result
    inside=mask.ravel().astype(bool);n=int(inside.sum());ratio=n/len(pairs)
    result.update(inliers=n,inlier_ratio=ratio)
    if n<limits['min_inliers'] or ratio<limits['min_inlier_ratio']:return result
    ca=float(cv2.contourArea(cv2.convexHull(qa[inside])))
    cb=float(cv2.contourArea(cv2.convexHull(qb[inside])))
    # Hull spans EMPTY space between isolated objects. Occupied cells add actual
    # spatial support; repeated SIFT orientations at one point count only once.
    occupied_a=len(np.unique(np.clip((qa[inside]*4).astype(int),0,3),axis=0))
    occupied_b=len(np.unique(np.clip((qb[inside]*4).astype(int),0,3),axis=0))
    unit=np.float32([[0,0],[1,0],[1,1],[0,1]])
    projected=cv2.perspectiveTransform(unit[None],H)[0]
    denominator=np.c_[unit,np.ones(4)]@H[2,:]
    area=float(cv2.contourArea(projected,oriented=True))
    if (not np.isfinite(projected).all() or min(abs(denominator))<1e-5
            or np.sign(denominator).min()!=np.sign(denominator).max()
            or not cv2.isContourConvex(projected) or not .05<area<20 or np.linalg.cond(H)>1e4):
        return result
    # Nearly coincident float32 polygons can yield zero intersection in OpenCV.
    # Use OpenCV's rasterized warp overlap, not a custom polygon-intersection implementation.
    grid=np.diag([256.,256.,1.]);pixel_H=grid@H@np.linalg.inv(grid)
    square=np.ones((256,256),np.uint8)
    overlap_view=float(cv2.warpPerspective(square,pixel_H,(256,256),flags=cv2.INTER_NEAREST).mean())
    overlap_query=float(cv2.warpPerspective(square,np.linalg.inv(pixel_H),(256,256),flags=cv2.INTER_NEAREST).mean())
    displacement=float(np.max(np.linalg.norm(projected-unit,axis=-1)))
    result.update(coverage_query=ca,coverage_view=cb,overlap_query=overlap_query,
                  overlap_view=overlap_view,corner_displacement=displacement,
                  occupied_query_cells=occupied_a,occupied_view_cells=occupied_b)
    # Broad spatial support avoids declaring a room identical from a small common object/person.
    identity=min(ca,cb)>=limits['min_hull'] and min(occupied_a,occupied_b)>=limits['min_cells']
    reusable=(identity and displacement<=limits['max_corner_move'] and min(overlap_query,overlap_view)>=limits['min_overlap']
              and abs(float(a['aspect'])/float(b['aspect'])-1)<=limits['max_aspect_change'])
    score=float(ratio*min(1,n/50)*min(1,min(ca,cb)/.35))
    result.update(identity=identity,reusable=reusable,score=score,homography=H.tolist())
    return result


def _dense_reuse_check(query,view,homography):
    """Additional appearance veto, not semantic/place identity. OpenCV NCC.

    Checks visible pixels OUTSIDE matched landmarks too. Low texture is unknown,
    not NCC=1 (OpenCV's constant-template special case). No new model/weights.
    """
    # Homography is in normalized SIFT-analysis pixel centers, not resized edges.
    # Align ORIGINAL pixels on the target's native canvas before decimation: a
    # downsampled wide view cannot recover crop detail by being warped larger.
    def native_grid(record):
        h,w=record.image.shape[:2];scale=min(1,640/max(h,w))
        fw,fh=max(1,round(w*scale)),max(1,round(h*scale))
        return np.array([[w,0,(w/fw-1)/2],[0,h,(h/fh-1)/2],[0,0,1.]])
    H=native_grid(view)@np.asarray(homography)@np.linalg.inv(native_grid(query))
    target=(view.image.shape[1],view.image.shape[0])
    def prepare(record,warp=None):
        image=rgb8(record.image);h,w=image.shape[:2]
        raw=np.ones((h,w),np.float32) if record.foreground_mask is None else (record.foreground_mask<.5).astype(np.float32)
        if warp is not None:
            image=cv2.warpPerspective(image,warp,target,flags=cv2.INTER_LINEAR)
            raw=cv2.warpPerspective(raw,warp,target,flags=cv2.INTER_LINEAR)
        image=cv2.GaussianBlur(cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA),(3,3),.8)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        valid=(cv2.resize(raw,(256,256),interpolation=cv2.INTER_AREA)>=1-1e-6).astype(np.uint8)
        return gray,cv2.erode(valid,np.ones((3,3),np.uint8)),image
    warped,qvalid,qrgb=prepare(query,H);v,vvalid,vrgb=prepare(view)
    joint=cv2.erode(((qvalid>0)&(vvalid>0)).astype(np.uint8),np.ones((3,3),np.uint8))
    scores=[];limits=DENSE_LIMITS;calibration=np.zeros_like(joint,dtype=bool)
    for y in range(0,256,64):
        for x in range(0,256,64):
            ok=joint[y:y+64,x:x+64]>0
            if ok.mean()<limits['min_cell_valid_fraction']:continue
            a=warped[y:y+64,x:x+64][ok].astype(np.float32).reshape(1,-1)
            b=v[y:y+64,x:x+64][ok].astype(np.float32).reshape(1,-1)
            if min(a.std(),b.std())<limits['min_std']:continue
            ncc=float(cv2.matchTemplate(a,b,cv2.TM_CCOEFF_NORMED)[0,0])
            if np.isfinite(ncc):
                scores.append(ncc)
                if ncc>=limits['min_ncc']:calibration[y:y+64,x:x+64]=ok
    agree=sum(s>=limits['min_ncc'] for s in scores)/max(1,len(scores))
    # NCC has no evidence about flat walls or RGB color. Fit only a global
    # exposure transform from agreed textured support; test ALL visible pixels,
    # including low-texture regions. Unknown regions never disappear from denominator.
    pixel_agree=0.;gain=offset=None
    if calibration.any():
        warped_rgb=qrgb.astype(np.float32)
        a,b=warped_rgb[calibration],vrgb[calibration].astype(np.float32)
        spread_a=np.diff(np.percentile(a,[10,90],axis=0),axis=0)[0]
        spread_b=np.diff(np.percentile(b,[10,90],axis=0),axis=0)[0]
        gain=np.clip(spread_b/np.maximum(spread_a,limits['min_std']),.5,2.)
        offset=np.median(b,axis=0)-gain*np.median(a,axis=0)
        residual=np.max(np.abs(warped_rgb*gain+offset-vrgb),axis=-1)
        pixel_agree=float(np.mean(residual[joint>0]<=limits['max_color_error']))
    return dict(method='warped-background-tile-NCC',textured_cells=len(scores),cell_scores=scores,
                agree_fraction=agree,pixel_agree_fraction=pixel_agree,
                exposure_gain=None if gain is None else gain.tolist(),exposure_offset=None if offset is None else offset.tolist(),
                passed=len(scores)>=limits['min_cells'] and agree>=limits['min_agree_fraction']
                       and pixel_agree>=limits['min_pixel_agree_fraction'],
                semantic_evidence=False)


def _matching_record(view):
    raw=view.candidate
    if (raw.foreground_mask is None and not raw.background_only and view.state=='READY'
            and view.reuse_exclusion_mask is not None):
        return replace(raw,foreground_mask=view.reuse_exclusion_mask)
    return raw


class EnvironmentMatcher:
    def match_for_production(self,candidate,views,revision='default'):
        """Bounded automatic dispatch, not a new physical-room classifier.

        Unproven/conflicting identities get an isolated PENDING template and a wash
        action. Never copy an old clean background just to avoid an extra wash.
        Persist revision across retries; change it only for deliberate correction.
        """
        key=production_request_key(candidate,revision)
        views=list(views);d=self.match(candidate,views)
        report={**d.report,'diagnostic_decision':d.decision,'diagnostic_reason':d.reason,
                'physical_identity_proven':False,'semantic_model_used':False,
                'automatic_fallback':d.decision=='UNCERTAIN'}
        if d.decision in ('REUSE_EXISTING','NEW_VIEW','UNCERTAIN'):
            # Local shared geometry alone cannot classify the physical environment.
            # Check all identity-supported histories, including non-reusable crops.
            # Different environments remain ambiguous unless dense evidence supports
            # exactly ONE environment. Equal/similar geometry scores never break ties.
            by_id={v.view_id:v for v in views};guards=[]
            choices=sorted((p for p in d.report['comparisons'] if p.get('identity')),
                           key=lambda p:(not p['reusable'],-p['score']))
            passed=[]
            for geom in choices:
                view=by_id[geom['view_id']]
                record=_matching_record(view)
                if geom['reusable']:
                    guard=_dense_reuse_check(candidate,record,geom['homography'])
                else:
                    # Test overlap on the current crop's canvas, not empty margins
                    # of the old wide view. Does not make the old template reusable.
                    guard=_dense_reuse_check(record,candidate,np.linalg.inv(geom['homography']))
                guards.append(dict(view_id=view.view_id,environment_id=view.environment_id,**guard))
                if guard['passed']:passed.append((geom,guards[-1]))
            report['dense_reuse_guards']=guards
            environments={p[0]['environment_id'] for p in passed}
            if len(environments)==1:
                geom,guard=passed[0]
                report['automatic_fallback']=False
                kind='REUSE_EXISTING' if geom['reusable'] else 'NEW_VIEW'
                d=replace(d,decision=kind,matched_environment_id=geom['environment_id'],
                          matched_view_id=geom['view_id'] if kind=='REUSE_EXISTING' else None,
                          global_similarity=geom['global_similarity'],geometry_score=geom['score'],confidence=geom['score'],
                          reason='unique background geometry + dense support; '+('safe view reuse' if kind=='REUSE_EXISTING' else 'new viewpoint'))
                report['dense_reuse_guard']=guard
            else:
                d=replace(d,decision='UNCERTAIN',matched_environment_id=None,matched_view_id=None,
                          reason='background consistency insufficient or supports multiple environments; isolate, do not group from geometry alone')
                if guards:report['dense_reuse_guard']=guards[-1]
                report['automatic_fallback']=True
        if d.decision=='UNCERTAIN':
            d=EnvironmentDecision(candidate.task_id,candidate.key,'NEW_ENVIRONMENT',
                                  reason='isolated template because existing clean reuse is unproven; not a physical new-room assertion')
        if d.decision=='REUSE_EXISTING':
            view=next(v for v in views if v.view_id==d.matched_view_id)
            report['production_action']='USE_READY_REFERENCE' if view.state=='READY' else 'ENSURE_WASH_EXISTING_VIEW'
        else:report['production_action']='WASH_NEW_VIEW'
        return replace(d,report=report,decision_key=key)

    def match(self,candidate,views):
        views=list(views)
        if any(v.task_id!=candidate.task_id for v in views):
            raise ValueError('all indexed views must belong to the same video task')
        report=dict(searched_views=len(views),comparisons=[],features_rebuilt=0,semantic_model_used=False,
                    confidence_is_calibrated_probability=False,geometry_limits=dict(GEOMETRY_LIMITS))
        def decision(kind,reason,view=None,global_score=None,geometry_score=None,confidence=0.):
            return EnvironmentDecision(candidate.task_id,candidate.key,kind,
                                       None if view is None else view.environment_id,
                                       view.view_id if view is not None and kind=='REUSE_EXISTING' else None,
                                       global_score,geometry_score,None,confidence,reason,report)
        if candidate.foreground_mask is None and not candidate.background_only:
            return decision('UNCERTAIN','foreground exclusion missing; raw person image is not proof of environment identity')
        query=extract_features(candidate)
        if float(query['visible'])<.1 or len(query['descriptors'])<GEOMETRY_LIMITS['min_inliers']:
            return decision('UNCERTAIN','insufficient visible/textured background for geometric identification')
        if not views:
            return decision('NEW_ENVIRONMENT','first usable candidate initializes this task cache; physical identity not inferred',confidence=1.)
        ranked=[]
        for view in views:
            record=_matching_record(view)
            if record.foreground_mask is None and not record.background_only:
                report['comparisons'].append(dict(view_id=view.view_id,reason='cached foreground exclusion missing'))
                continue
            features=view.reuse_features if record is not view.candidate else view.features
            if not _valid_features(features,record):
                features=extract_features(record)
                report['features_rebuilt']+=1
            similarity=float(np.clip(1-cv2.compareHist(query['histogram'],features['histogram'],cv2.HISTCMP_BHATTACHARYYA),0,1))
            ranked.append((similarity,view,features))
        # Global sorting is cheap and used for prioritization, NEVER sole same-room decision.
        matches=[]
        for similarity,view,features in sorted(ranked,key=lambda p:p[0],reverse=True):
            geom=_geometry(query,features)
            report['comparisons'].append(dict(view_id=view.view_id,environment_id=view.environment_id,
                                              global_similarity=similarity,**geom))
            if geom['identity']:matches.append((geom['score'],similarity,view,geom))
        if not matches:
            return decision('UNCERTAIN','no verified shared background geometry; absence of a match does not prove a new physical environment',
                            global_score=max((p[0] for p in ranked),default=0.))
        matches.sort(key=lambda p:p[0],reverse=True)
        best=matches[0]
        if any(other[2].environment_id!=best[2].environment_id for other in matches[1:]):
            return decision('UNCERTAIN','ambiguous geometry supports multiple cached environments',
                            global_score=best[1],geometry_score=best[0])
        # Prefer a geometrically reusable view of the selected environment over its other large-view matches.
        usable=[p for p in matches if p[2].environment_id==best[2].environment_id and p[3]['reusable']]
        selected=max(usable,key=lambda p:p[0]) if usable else best
        kind='REUSE_EXISTING' if usable else 'NEW_VIEW'
        return decision(kind,'shared distributed background structure; '+('nearby view suitable for reuse' if usable else 'viewpoint/crop differs too much for reuse'),
                        selected[2],selected[1],selected[0],selected[0])

