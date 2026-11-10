"""
Perception metrics against ground truth.

Per frame: greedy nearest matching of detections to true cones (car frame)
within `gate` metres -> TP / FP / FN, position errors, colour confusion.
Recall is measured against the cones that were *reachable* (inside
`eval_range` and in front of the car's rear axle) so that a narrow camera
isn't credited for cones behind it, and is also binned by range so the
"recall vs range" curve shows where a sensor runs out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..core.types import N_COLORS, UNKNOWN

RANGE_BINS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def match(det_xy, gt_xy, gate=0.75):
    """Greedy nearest pairs. Returns (det_idx, gt_idx) arrays."""
    if not len(det_xy) or not len(gt_xy):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    d = np.hypot(det_xy[:, None, 0] - gt_xy[None, :, 0], det_xy[:, None, 1] - gt_xy[None, :, 1])
    di, gi = [], []
    flat = np.argsort(d, axis=None)
    used_d, used_g = set(), set()
    for f in flat:
        i, j = divmod(int(f), d.shape[1])
        if d[i, j] > gate:
            break
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        di.append(i)
        gi.append(j)
    return np.array(di, dtype=int), np.array(gi, dtype=int)


@dataclass
class PerceptionStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sq_err: float = 0.0
    err_x: float = 0.0
    err_y: float = 0.0
    color_confusion: np.ndarray = field(default_factory=lambda: np.zeros((N_COLORS, N_COLORS), dtype=int))
    bin_tp: np.ndarray = field(default_factory=lambda: np.zeros(len(RANGE_BINS) - 1, dtype=int))
    bin_gt: np.ndarray = field(default_factory=lambda: np.zeros(len(RANGE_BINS) - 1, dtype=int))
    frames: int = 0

    def add(self, det, gt_car, eval_range=30.0, gate=0.75):
        """det: (N, 4) car frame; gt_car: (M, 4) true cones in the car frame."""
        self.frames += 1
        r_gt = np.hypot(gt_car[:, 0], gt_car[:, 1])
        reach = (r_gt <= eval_range) & (gt_car[:, 0] > -0.5)
        gt = gt_car[reach]
        r_gt = r_gt[reach]
        # judge detections in the same region (a 360 deg LiDAR seeing cones
        # behind the car is right, not a false positive); keep a margin so
        # a detection just over the boundary still matches its cone
        if len(det):
            r_det = np.hypot(det[:, 0], det[:, 1])
            det = det[(r_det <= eval_range + 1.0) & (det[:, 0] > -1.5)]
        di, gi = match(det[:, :2], gt[:, :2], gate)
        self.tp += len(di)
        self.fp += len(det) - len(di)
        self.fn += len(gt) - len(gi)
        if len(di):
            e = det[di, :2] - gt[gi, :2]
            self.sq_err += float((e ** 2).sum())
            self.err_x += float(e[:, 0].sum())
            self.err_y += float(e[:, 1].sum())
            for c_true, c_det in zip(gt[gi, 2].astype(int), det[di, 2].astype(int)):
                self.color_confusion[c_true, c_det] += 1
        b_all = np.clip(np.digitize(r_gt, RANGE_BINS) - 1, 0, len(RANGE_BINS) - 2)
        self.bin_gt += np.bincount(b_all, minlength=len(RANGE_BINS) - 1)
        if len(gi):
            self.bin_tp += np.bincount(b_all[gi], minlength=len(RANGE_BINS) - 1)
        return di, gi

    def merge(self, other):
        for k in ('tp', 'fp', 'fn', 'sq_err', 'err_x', 'err_y', 'frames'):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.color_confusion += other.color_confusion
        self.bin_tp += other.bin_tp
        self.bin_gt += other.bin_gt

    @property
    def precision(self):
        return self.tp / max(1, self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / max(1, self.tp + self.fn)

    @property
    def rmse(self):
        return float(np.sqrt(self.sq_err / max(1, self.tp)))

    @property
    def color_accuracy(self):
        """Correct colour among matched cones that were given a colour."""
        cm = self.color_confusion
        coloured = cm[:, :UNKNOWN].sum()
        return float(np.trace(cm[:UNKNOWN, :UNKNOWN]) / coloured) if coloured else float('nan')

    @property
    def colored_fraction(self):
        cm = self.color_confusion
        tot = cm.sum()
        return float(cm[:, :UNKNOWN].sum() / tot) if tot else 0.0

    def recall_by_range(self):
        return np.where(self.bin_gt > 0, self.bin_tp / np.maximum(self.bin_gt, 1), np.nan)

    def summary(self) -> Dict:
        return {'frames': self.frames, 'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
                'precision': round(self.precision, 4), 'recall': round(self.recall, 4),
                'rmse_m': round(self.rmse, 4), 'bias_x_m': round(self.err_x / max(1, self.tp), 4),
                'bias_y_m': round(self.err_y / max(1, self.tp), 4),
                'color_accuracy': None if np.isnan(self.color_accuracy) else round(self.color_accuracy, 4),
                'colored_fraction': round(self.colored_fraction, 4),
                'recall_by_range': [None if np.isnan(v) else round(float(v), 3) for v in self.recall_by_range()],
                'range_bins': list(RANGE_BINS)}


def range_labels() -> List[str]:
    return ['{:g}-{:g}'.format(a, b) for a, b in zip(RANGE_BINS[:-1], RANGE_BINS[1:])]
