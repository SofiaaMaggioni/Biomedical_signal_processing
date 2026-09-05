"""Evaluation helpers for QRS detection performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QRSEvaluationResult:
    """Summary of detected QRS positions compared with reference positions."""

    true_positives: int  # Detected beats matched to a reference beat.
    false_positives: int  # Detected beats without a matching reference beat.
    false_negatives: int  # Reference beats missed by the detector.
    sensitivity: float  # TP / (TP + FN).
    positive_predictive_value: float  # TP / (TP + FP).
    f1_score: float  # Harmonic mean of sensitivity and PPV.
    mean_abs_error_ms: float  # Mean absolute timing error of matched beats.
    tolerance_s: float  # Matching tolerance in seconds.
    matches: np.ndarray  # Columns: reference sample, detected sample.
    false_positive_samples: np.ndarray  # Detected samples not matched to reference.
    false_negative_samples: np.ndarray  # Reference samples not matched to detection.


def match_qrs_detections(
    detected_samples: np.ndarray,
    reference_samples: np.ndarray,
    fs: float,
    tolerance_s: float = 0.05,
) -> QRSEvaluationResult:
    """Match detected QRS positions to reference positions.

    A detected beat is counted as correct when it is within ``tolerance_s`` of
    one reference beat. Each detected beat can match at most one reference beat.
    """

    detected = np.asarray(detected_samples, dtype=int)
    reference = np.asarray(reference_samples, dtype=int)
    if detected.ndim != 1 or reference.ndim != 1:
        raise ValueError("detected_samples and reference_samples must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")

    detected = np.sort(detected)
    reference = np.sort(reference)
    tolerance_samples = int(round(tolerance_s * fs))

    used_detected = np.zeros(len(detected), dtype=bool)
    matched_pairs: list[tuple[int, int]] = []
    missed_reference: list[int] = []

    for ref_sample in reference:
        if len(detected) == 0:
            missed_reference.append(int(ref_sample))
            continue

        distances = np.abs(detected - ref_sample)
        distances[used_detected] = np.iinfo(np.int64).max
        best_idx = int(np.argmin(distances))

        if distances[best_idx] <= tolerance_samples:
            used_detected[best_idx] = True
            matched_pairs.append((int(ref_sample), int(detected[best_idx])))
        else:
            missed_reference.append(int(ref_sample))

    false_positive_samples = detected[~used_detected]
    matches = np.asarray(matched_pairs, dtype=int)
    false_negative_samples = np.asarray(missed_reference, dtype=int)

    tp = len(matches)
    fp = len(false_positive_samples)
    fn = len(false_negative_samples)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    if sensitivity + ppv > 0:
        f1_score = 2 * sensitivity * ppv / (sensitivity + ppv)
    else:
        f1_score = np.nan

    if len(matches) > 0:
        timing_errors_ms = (matches[:, 1] - matches[:, 0]) / fs * 1000.0
        mean_abs_error_ms = float(np.mean(np.abs(timing_errors_ms)))
    else:
        mean_abs_error_ms = np.nan

    return QRSEvaluationResult(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        sensitivity=float(sensitivity),
        positive_predictive_value=float(ppv),
        f1_score=float(f1_score),
        mean_abs_error_ms=mean_abs_error_ms,
        tolerance_s=tolerance_s,
        matches=matches,
        false_positive_samples=false_positive_samples.astype(int),
        false_negative_samples=false_negative_samples.astype(int),
    )
