import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from yolox.tracker import kalman_filter
import torch
import torch.nn as nn
import math
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def dious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    dious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if dious.size == 0:
        return dious

    pred = np.ascontiguousarray(atlbrs, dtype=np.float)
    target = np.ascontiguousarray(btlbrs, dtype=np.float)
    eps = 1e-7
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    lt = torch.max(pred[..., :, None, :2], target[..., None, :, :2])
    rb = torch.min(pred[..., :, None, 2:], target[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    # union
    ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    ag = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    union = ap[..., None] + ag[..., None, :] - overlap
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :, None, :2], target[..., None, :, :2])
    enclose_x2y2 = torch.max(pred[..., :, None, 2:], target[..., None, :, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[..., 0]
    ch = enclose_wh[..., 1]

    c2 = cw**2 + ch**2
    c2 = torch.max(c2, eps)

    b1_x1, b1_y1 = pred[..., 0], pred[..., 1]
    b1_x2, b1_y2 = pred[..., 2], pred[..., 3]
    b2_x1, b2_y1 = target[..., 0], target[..., 1]
    b2_x2, b2_y2 = target[..., 2], target[..., 3]

    left = ((b2_x1 + b2_x2)[..., None, :] - (b1_x1 + b1_x2)[..., None])**2 / 4
    right = ((b2_y1 + b2_y2)[..., None, :] - (b1_y1 + b1_y2)[..., None])**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    dious = dious.numpy()

    return dious


def eious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    eious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if eious.size == 0:
        return eious

    pred = np.ascontiguousarray(atlbrs, dtype=np.float)
    target = np.ascontiguousarray(btlbrs, dtype=np.float)
    eps = 1e-7
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)

    lt = torch.max(pred[..., :, None, :2], target[..., None, :, :2])
    rb = torch.min(pred[..., :, None, 2:], target[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    # union
    ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    ag = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    union = ap[..., None] + ag[..., None, :] - overlap
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :, None, :2], target[..., None, :, :2])
    enclose_x2y2 = torch.max(pred[..., :, None, 2:], target[..., None, :, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[..., 0]
    ch = enclose_wh[..., 1]

    c2 = cw**2 + ch**2
    c2 = torch.max(c2, eps)

    b1_x1, b1_y1 = pred[..., 0], pred[..., 1]
    b1_x2, b1_y2 = pred[..., 2], pred[..., 3]
    b2_x1, b2_y1 = target[..., 0], target[..., 1]
    b2_x2, b2_y2 = target[..., 2], target[..., 3]

    left = ((b2_x1 + b2_x2)[..., None, :] - (b1_x1 + b1_x2)[..., None])**2 / 4
    right = ((b2_y1 + b2_y2)[..., None, :] - (b1_y1 + b1_y2)[..., None])**2 / 4
    rho2 = left + right
    rho_w2 = ((b2_x2 - b2_x1)[..., None, :] - (b1_x2 - b1_x1)[..., None]) ** 2
    rho_h2 = ((b2_y2 - b2_y1)[..., None, :] - (b1_y2 - b1_y1)[..., None]) ** 2
    cw2 = cw ** 2
    cw2 = torch.max(cw2, eps)
    ch2 = ch ** 2
    ch2 = torch.max(ch2, eps)

    # eIoU
    eious = ious - rho2 / c2 - rho_h2/ch2 - rho_w2/cw2
    eious = eious.numpy()

    return eious


def cious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    cious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if cious.size == 0:
        return cious

    pred = np.ascontiguousarray(atlbrs, dtype=np.float)
    target = np.ascontiguousarray(btlbrs, dtype=np.float)
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    eps = 1e-7
    lt = torch.max(pred[..., :, None, :2], target[..., None, :, :2])
    rb = torch.min(pred[..., :, None, 2:], target[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]

    # union
    ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    ag = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
    union = ap[..., None] + ag[..., None, :] - overlap
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[..., :, None, :2], target[..., None, :, :2])
    enclose_x2y2 = torch.max(pred[..., :, None, 2:], target[..., None, :, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[..., 0]
    ch = enclose_wh[..., 1]

    c2 = cw ** 2 + ch ** 2
    c2 = torch.max(c2, eps)

    b1_x1, b1_y1 = pred[..., 0], pred[..., 1]
    b1_x2, b1_y2 = pred[..., 2], pred[..., 3]
    b2_x1, b2_y1 = target[..., 0], target[..., 1]
    b2_x2, b2_y2 = target[..., 2], target[..., 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2)[..., None, :] - (b1_x1 + b1_x2)[..., None]) ** 2 / 4
    right = ((b2_y1 + b2_y2)[..., None, :] - (b1_y1 + b1_y2)[..., None]) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2)[..., None, :] - torch.atan(w1 / h1)[..., None], 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    cious = cious.numpy()

    return cious

def gious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    gious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if gious.size == 0:
        return gious
    pred = np.array(atlbrs, dtype=np.float)
    target = np.array(btlbrs, dtype=np.float)
    eps = 1e-7
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    gious = bbox_overlaps1(pred, target, mode='giou', is_aligned=False, eps=eps)
    # # lt = torch.max(pred[..., :, None, :2], target[..., None, :, :2])
    # # rb = torch.min(pred[..., :, None, 2:], target[..., None, :, 2:])
    # lt = np.maximum(pred[..., :, None, :2], target[..., None, :, :2])
    # rb = np.minimum(pred[..., :, None, 2:], target[..., None, :, 2:])
    # wh = np.maximum(0., rb - lt)
    # overlap = wh[:, 0] * wh[:, 1]
    #
    # # union
    # ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    # ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    # union = ap[..., None] + ag[..., None, :] - overlap + eps
    #
    # # IoU
    # ious = overlap / union
    #
    # # enclose area
    # # enclose_x1y1 = torch.min(pred[..., :, None, :2], target[..., None, :, :2])
    # # enclose_x2y2 = torch.max(pred[..., :, None, 2:], target[..., None, :, 2:])
    # enclose_x1y1 = np.minimum(pred[:, :2], target[:, :2])
    # enclose_x2y2 = np.maximum(pred[:, 2:], target[:, 2:])
    # enclose_wh = np.maximum(0., enclose_x2y2 - enclose_x1y1)
    #
    # cw = enclose_wh[:, 0]
    # ch = enclose_wh[:, 1]
    # enclose_area = cw * ch
    # enclose_area = np.maximum(enclose_area, eps)
    # gious = ious - (enclose_area - union) / enclose_area
    #
    gious = gious.numpy()
    return gious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def giou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = eious(atlbrs, btlbrs)                           # giou diou ciou
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def bbox_overlaps1(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned `` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious
