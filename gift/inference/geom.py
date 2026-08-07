import cadquery as cq
import numpy as np
from typing import Tuple, Union

def _compute_iou_centering(source : cq.Workplane, target : cq.Workplane) -> float:
    """Compute IOU after centering both shapes at the origin (no rotation or scale normalization)."""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    centered_source = source.translate(-c_source).val()
    centered_target = target.translate(-c_target).val()

    try:
        intersect = centered_source.intersect(centered_target)
        union = centered_source.fuse(centered_target)
        return intersect.Volume() / union.Volume()
    except:
        return 0.0


def _compute_iou_centering_normalize(source : cq.Workplane, target : cq.Workplane) -> float:
    """Compute IOU after centering and scale-normalizing both shapes (no rotation)."""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, _ = np.linalg.eigh(I_source)
    I_p_target, _ = np.linalg.eigh(I_target)

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    try:
        intersect = normalized_source.intersect(normalized_target)
        union = normalized_source.fuse(normalized_target)
        return intersect.Volume() / union.Volume()
    except:
        return 0.0


def align_shapes(source : cq.Workplane, target : cq.Workplane) -> Tuple[cq.Workplane, float]:
    """Align source to target using the center of mass and the principal axes of inertia. also return normalized IOU"""
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())

    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))

    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())

    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)

    s_source = np.sqrt(np.abs(I_p_source).sum()/v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum()/v_target)

    normalized_source = source.translate(-c_source).val().scale(1/s_source)
    normalized_target = target.translate(-c_target).val().scale(1/s_target)

    Rs = np.zeros((4,3,3))
    Rs[0] = I_v_target @ I_v_source.T

    for i in range(3):
        # all possible 2 out of 3 permutations
        alignment = 1 - 2 * np.array([i>0, (i+1)%2, i%3<=1])
        Rs[i+1] = I_v_target @ (alignment[None,:] * I_v_source).T

    best_IOU = 0.0
    best_T = None
    for i in range(4):
        T = np.zeros([4,4])
        T[:3,:3] = Rs[i]
        T[-1,-1] = 1
        
        
        try:
            aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            
            IOU = intersect.Volume() / union.Volume()
        except: #handle cases where IOU is undefined
            IOU = 0.0
        
        if IOU > best_IOU:
            best_IOU = IOU
            best_T = T

    if best_T is not None:
        aligned_source = normalized_source.transformGeometry(cq.Matrix(best_T.tolist())).scale(s_target).translate(c_target)
        return cq.Workplane(aligned_source), best_IOU
    else:
        aligned_source = None
        return aligned_source, best_IOU