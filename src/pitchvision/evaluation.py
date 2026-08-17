"""Validation metrics for the pipeline's three main sources of error: YOLO
detection, homography pitch calibration, and K-means team-colour
clustering. None of these has a pre-existing ground-truth split for this
project's own broadcast footage, so every function here is meant to be
scored against a small hand-labeled sample (a handful of frames' boxes, a
few extra calibration landmarks, a few tracks' true team) rather than a
full benchmark dataset - enough to report a number, not to replace a
proper evaluation set.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .calibration import PitchCalibrator


# ---------------------------------------------------------------------------
# 1. Detection accuracy (YOLO)
# ---------------------------------------------------------------------------

_BBOX_COLUMNS = ["frame", "class_name", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def predicted_boxes_dataframe(frame_detections: Dict[int, Sequence]) -> pd.DataFrame:
    """Builds a `detection_metrics`-shaped DataFrame from
    `{frame_index: [Detection, ...]}`, as returned by
    `PlayerBallDetector.detect` on a handful of individually-chosen frames -
    NOT a full tracked clip, since validating against hand-labeled ground
    truth only needs the specific frames that were labeled."""
    rows = []
    for frame_idx, detections in frame_detections.items():
        for d in detections:
            x1, y1, x2, y2 = d.xyxy
            rows.append(
                {"frame": frame_idx, "class_name": d.class_name, "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2}
            )
    return pd.DataFrame(rows, columns=_BBOX_COLUMNS)


def ground_truth_boxes_dataframe(
    frame_boxes: Dict[int, Sequence[Tuple[str, float, float, float, float]]]
) -> pd.DataFrame:
    """Builds the matching ground-truth DataFrame from
    `{frame_index: [(class_name, x1, y1, x2, y2), ...]}` - hand-typed boxes
    read off a hover-enabled frame display (`plotly.express.imshow`),
    mirroring the manual pitch-landmark fallback pattern used for
    calibration in notebook 00."""
    rows = []
    for frame_idx, boxes in frame_boxes.items():
        for class_name, x1, y1, x2, y2 in boxes:
            rows.append(
                {"frame": frame_idx, "class_name": class_name, "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2}
            )
    return pd.DataFrame(rows, columns=_BBOX_COLUMNS)


def _match_frame_class(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float):
    """Greedy highest-IoU-first matching between predicted and ground-truth
    boxes of ONE class in ONE frame. Returns (tp, fp, fn, matched_ious)."""
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0:
        return 0, 0, n_gt, []
    if n_gt == 0:
        return 0, n_pred, 0, []

    pairs = []
    for i in range(n_pred):
        for j in range(n_gt):
            iou = box_iou(pred_boxes[i], gt_boxes[j])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)

    matched_pred, matched_gt, matched_ious = set(), set(), []
    for iou, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matched_ious.append(iou)

    tp = len(matched_ious)
    return tp, n_pred - tp, n_gt - tp, matched_ious


def compute_detection_metrics(
    predicted: pd.DataFrame, ground_truth: pd.DataFrame, iou_threshold: float = 0.5
) -> pd.DataFrame:
    """Precision/recall/F1/mean-IoU, matched independently per (frame,
    class), between `predicted` (e.g. `predicted_boxes_dataframe`, or a
    filtered slice of `TrackingPipeline.run()`'s output) and `ground_truth`
    (`ground_truth_boxes_dataframe`, hand-labeled for the SAME frames) -
    both need `frame`, `class_name`, `bbox_x1`, `bbox_y1`, `bbox_x2`,
    `bbox_y2` columns at minimum.

    Matching is class-scoped: a correctly-placed box with the wrong class
    label counts as a false positive for its predicted class AND a false
    negative for its true class, rather than a match - a mislabeled
    detection is still an error worth counting. Returns one row per class
    plus an "overall" row (micro-averaged: totals pooled across classes
    before computing precision/recall, so classes with more instances
    weigh proportionally more)."""
    classes = sorted(set(predicted["class_name"]) | set(ground_truth["class_name"]))
    frames = sorted(set(predicted["frame"]) | set(ground_truth["frame"]))

    totals = {cls: {"tp": 0, "fp": 0, "fn": 0, "ious": []} for cls in classes}
    for frame in frames:
        pred_frame = predicted[predicted["frame"] == frame]
        gt_frame = ground_truth[ground_truth["frame"] == frame]
        for cls in classes:
            pred_boxes = pred_frame.loc[pred_frame["class_name"] == cls, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
            gt_boxes = gt_frame.loc[gt_frame["class_name"] == cls, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy()
            tp, fp, fn, ious = _match_frame_class(pred_boxes, gt_boxes, iou_threshold)
            totals[cls]["tp"] += tp
            totals[cls]["fp"] += fp
            totals[cls]["fn"] += fn
            totals[cls]["ious"].extend(ious)

    def _row(name: str, tp: int, fp: int, fn: int, ious: list) -> dict:
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
        return {
            "class_name": name,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        }

    rows = [_row(cls, t["tp"], t["fp"], t["fn"], t["ious"]) for cls, t in totals.items()]
    rows.append(
        _row(
            "overall",
            sum(t["tp"] for t in totals.values()),
            sum(t["fp"] for t in totals.values()),
            sum(t["fn"] for t in totals.values()),
            [iou for t in totals.values() for iou in t["ious"]],
        )
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Homography calibration accuracy
# ---------------------------------------------------------------------------


def compute_calibration_holdout_error(
    pixel_points: np.ndarray,
    pitch_points: np.ndarray,
    n_fit: Optional[int] = None,
    n_repeats: int = 20,
    seed: int = 0,
) -> dict:
    """Repeated random-subsampling (Monte Carlo) validation of calibration
    accuracy: fits a homography on a random subset of `n_fit` of the given
    pixel<->pitch correspondences and measures reprojection error (metres)
    on the REST, repeating with a different random split `n_repeats` times
    and pooling every holdout error. `PitchCalibrator.reprojection_error`
    alone reports in-sample error - measured on the exact points the
    homography was fit from, which always looks good and doesn't reflect
    accuracy elsewhere on the pitch; this instead needs more landmark
    correspondences than the >=4 a fit requires (e.g. 10-15, hand-picked the
    same way as the manual fallback in `calibration.py`) so some can always
    be held out.

    Returns a dict with `n_points`, `n_fit`, `n_repeats`, `mean_error_m`,
    `std_error_m`, `max_error_m`, and `holdout_errors_m` (every pooled
    holdout error, for plotting a distribution)."""
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    pitch_points = np.asarray(pitch_points, dtype=np.float64)
    n = len(pixel_points)
    if n != len(pitch_points):
        raise ValueError("pixel_points and pitch_points must be the same length.")
    if n_fit is None:
        n_fit = max(4, n // 2)
    if n_fit < 4:
        raise ValueError("n_fit must be >= 4 (a homography needs >= 4 correspondences).")
    if n_fit >= n:
        raise ValueError(f"n_fit ({n_fit}) must leave at least 1 point held out (n={n}).")

    rng = np.random.default_rng(seed)
    holdout_errors = []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        fit_idx, holdout_idx = perm[:n_fit], perm[n_fit:]
        calibrator = PitchCalibrator.from_point_pairs(pixel_points[fit_idx], pitch_points[fit_idx])
        predicted = calibrator.pixel_to_pitch(pixel_points[holdout_idx])
        errors = np.linalg.norm(predicted - pitch_points[holdout_idx], axis=1)
        holdout_errors.extend(errors.tolist())

    holdout_errors = np.array(holdout_errors)
    return {
        "n_points": n,
        "n_fit": n_fit,
        "n_repeats": n_repeats,
        "mean_error_m": float(holdout_errors.mean()),
        "std_error_m": float(holdout_errors.std()),
        "max_error_m": float(holdout_errors.max()),
        "holdout_errors_m": holdout_errors,
    }


# ---------------------------------------------------------------------------
# 3. Team-colour clustering accuracy
# ---------------------------------------------------------------------------


def compute_clustering_accuracy(true_labels: Sequence, predicted_labels: Sequence) -> dict:
    """Accuracy of `predicted_labels` (KMeans cluster ids) against
    `true_labels` (hand-labeled real team identity for a sample of tracks),
    under the OPTIMAL cluster-id-to-true-label matching (the Hungarian
    algorithm on the confusion matrix, `scipy.optimize.linear_sum_assignment`
    - the same tool `set_pieces.match_formations` uses for a different
    optimal-pairing problem) - required because a cluster id (0/1) carries
    no identity of its own and isn't guaranteed to line up with any
    particular label numbering. Both label sequences must exclude anything
    that shouldn't count towards team accuracy (e.g. `team_id is None`
    referees) before calling this.

    Returns a dict with `accuracy`, `n_samples`, `confusion_matrix` (a
    DataFrame, rows = true labels, columns = predicted cluster ids), and
    `cluster_to_true_label` (the matching found)."""
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must be the same length.")

    true_classes = np.unique(true_labels)
    pred_classes = np.unique(predicted_labels)
    confusion = np.array(
        [[np.sum((true_labels == t) & (predicted_labels == p)) for p in pred_classes] for t in true_classes]
    )

    row_idx, col_idx = linear_sum_assignment(-confusion)
    accuracy = confusion[row_idx, col_idx].sum() / len(true_labels)
    cluster_to_true_label = {pred_classes[c]: true_classes[r] for r, c in zip(row_idx, col_idx)}

    return {
        "accuracy": float(accuracy),
        "n_samples": len(true_labels),
        "confusion_matrix": pd.DataFrame(confusion, index=true_classes, columns=pred_classes),
        "cluster_to_true_label": cluster_to_true_label,
    }
