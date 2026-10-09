"""One candidate per RAW Shot. OpenCV quality/motion primitives + H3 scoring glue.

Katna's quality-first selection informed the design, not copied code.
No sampling, padding, segmentation or background washing occurs here.
"""
import cv2
import numpy as np
try:
    from .h3_background_contracts import CandidateBackgroundRecord, require_mask, rgb8
except ImportError:
    from h3_background_contracts import CandidateBackgroundRecord, require_mask, rgb8


def _score(sharpness,exposure,visible,motion):
    sharp=float(np.clip(np.log1p(sharpness)/np.log1p(1000),0,1))
    stability=.5 if motion is None else 1/(1+motion/.005)
    return float((.35*sharp+.2*exposure+.35*visible+.1*stability)*visible)


def select_best_frame(frames,task_id,shot_number,person_masks=None,depth=None,analysis_edge=320,
                      foreground_provider=None,candidate_limit=5,background_only=False):
    if not isinstance(frames,np.ndarray) or frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] != 3:
        raise ValueError('frames must be non-empty raw Shot FHWC RGB')
    n,h,w,_=frames.shape
    if not isinstance(background_only,bool):raise ValueError('background_only must be an explicit bool')
    if background_only and foreground_provider is not None:
        raise ValueError('confirmed background-only Shot must not also request unknown foreground detection')
    if min(h,w)<1 or isinstance(analysis_edge,bool) or not isinstance(analysis_edge,int) or analysis_edge<16:
        raise ValueError('invalid frame dimensions/analysis_edge')
    if person_masks is not None:
        if not isinstance(person_masks,np.ndarray) or person_masks.shape != frames.shape[:3]:
            raise ValueError('person_masks must be aligned FHW, foreground=1')
        for mask in person_masks:require_mask(mask,(h,w))
        if background_only and np.any(person_masks):
            raise ValueError('background_only contradicts supplied foreground masks')
    if depth is not None and (not isinstance(depth,np.ndarray) or depth.ndim not in (3,4)
                              or depth.shape[:3] != frames.shape[:3] or depth.size==0
                              or depth.dtype.kind not in 'ufi' or not np.isfinite(depth).all()):
        raise ValueError('Depth must preserve the same raw Shot F/H/W')
    if foreground_provider is not None and not callable(foreground_provider):
        raise ValueError('foreground_provider must be callable(image,index) -> H/W mask or None')
    if isinstance(candidate_limit,bool) or not isinstance(candidate_limit,int) or candidate_limit<1:
        raise ValueError('candidate_limit must be a positive integer')
    scale=min(1.,analysis_edge/max(h,w));size=(max(1,round(w*scale)),max(1,round(h*scale)))
    metrics=[]; previous=None; previous_valid=None
    for i,frame in enumerate(frames):
        small=cv2.resize(rgb8(frame),size,interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
        raw_visible=np.ones((h,w),np.float32) if person_masks is None else (person_masks[i]<.5).astype(np.float32)
        # Match AREA RGB's whole resampling footprint; NEAREST can erase thin foreground.
        visible=(cv2.resize(raw_visible,size,interpolation=cv2.INTER_AREA)>=1-1e-6).astype(np.uint8)
        valid=cv2.erode(visible,np.ones((3,3),np.uint8)) if min(gray.shape)>=5 else visible
        pixels=gray[valid>0]; visible_fraction=float(np.mean(raw_visible))
        sharpness=float(cv2.Laplacian(gray,cv2.CV_32F)[valid>0].var()) if pixels.size else 0.
        exposure=float(np.mean((pixels>10)&(pixels<245))) if pixels.size else 0.
        motion=None
        if previous is not None and min(gray.shape)>=8:
            joint=(valid>0)&(previous_valid>0)
            if joint.sum()>=64 and gray[joint].var()>4 and previous[joint].var()>4:
                flow=cv2.calcOpticalFlowFarneback(previous,gray,None,.5,2,15,2,5,1.2,0)
                motion=float(np.median(np.linalg.norm(flow[joint],axis=-1)))/np.hypot(*gray.shape)
        # Explicit business weights, NOT a trained/calibrated quality classifier.
        score=_score(sharpness,exposure,visible_fraction,motion)
        if n>=5 and i in (0,n-1):score*=.95  # modest cut-edge penalty, never deletes real frames
        metrics.append(dict(index=i,sharpness=sharpness,exposure=exposure,
                            visible_fraction=None if person_masks is None and not background_only else visible_fraction,
                            motion=motion,score=float(score),visibility_source=('confirmed_background_only' if background_only else
                                'unknown' if person_masks is None else 'supplied_mask'),
                            motion_support=('confirmed_background_only' if background_only else
                                'whole_RGB_foreground_unknown' if person_masks is None else 'masked_background')))
        previous,previous_valid=gray,valid
    # Stable ties: retain earliest equal-quality valid frame; no arbitrary middle-frame rule.
    best=max(range(n),key=lambda i:metrics[i]['score']); selected_mask=None
    examined=[]
    if person_masks is None and foreground_provider is not None:
        # Only a bounded shortlist invokes the existing external detector/segmenter.
        # Temporal coverage avoids selecting exclusively similar sharp occluder frames.
        limit=min(n,candidate_limit)
        ranked=sorted(range(n),key=lambda i:(-metrics[i]['score'],i))
        shortlist=ranked[:max(1,(limit+1)//2)]
        for i in np.linspace(0,n-1,limit,dtype=int).tolist()+ranked:
            if len(shortlist)>=limit:break
            if i not in shortlist:shortlist.append(i)
        masks={}
        for i in shortlist:
            image=frames[i].view();image.setflags(write=False)
            mask=foreground_provider(image,i);examined.append(i)
            if mask is None:continue  # No detection is UNKNOWN, never proof of an empty scene.
            require_mask(mask,(h,w));masks[i]=mask.copy()
        for i,mask in masks.items():
            start=i-1 if i>0 and i-1 in masks else i
            aligned=np.stack([masks[j] for j in range(start,i+1)])
            refined=select_best_frame(frames[start:i+1],task_id,shot_number,person_masks=aligned,analysis_edge=analysis_edge)
            item=dict(refined.report['frames'][-1]);item.update(index=i,visibility_source='candidate_provider')
            if n>=5 and i in (0,n-1):item['score']*=.95
            metrics[i]=item
        if masks:
            best=max(masks,key=lambda i:(metrics[i]['score'],-i));selected_mask=masks[best]
    limitations=[]
    if person_masks is None and not background_only:limitations.append('person_mask_missing')
    if metrics[best]['visible_fraction'] is None:limitations.append('foreground_occupancy_unknown')
    if metrics[best]['visible_fraction']==0:limitations.append('no_visible_background')
    if any(m['motion'] is None for m in metrics):limitations.append('motion_unknown')
    report=dict(raw_shot_frames=n,analysis_size=list(size),selected_index=best,frames=metrics,
                limitations=limitations,quality_is_heuristic=True,
                motion_definition='incoming optical-flow median / analysis diagonal; masked background only when both masks exist, otherwise whole RGB; None is unobserved',
                motion_unknown_stability_prior=.5,
                candidate_refinement_indices=examined,
                visibility_unknown_scoring_prior=1.,background_only=background_only,
                cut_edge_policy='5% score penalty on first/last only when F>=5',
                not_implemented=['person_removal','semantic_structure_completeness','neural_quality_prediction'])
    return CandidateBackgroundRecord(task_id,shot_number,best,frames[best],metrics[best]['score'],
                                     selected_mask if person_masks is None else person_masks[best],
                                     None if depth is None else depth[best],background_only,report)

