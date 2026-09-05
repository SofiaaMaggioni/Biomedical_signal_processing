"""Reusable end-to-end fetal ECG pipeline.

The notebooks explain every step visually. This module keeps the same steps in
one callable function so we can run the algorithm on multiple records without
copying notebook code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_io import FetalEcgRecord, load_record
from evaluation import QRSEvaluationResult, match_qrs_detections
from fecg_averaging import FECGAverageResult, average_fetal_ecg
from mecg_cancellation import MECGCancellationResult, cancel_maternal_ecg
from preprocessing import (
    BaselineRemovalResult,
    PowerlineRemovalResult,
    UpsamplingResult,
    remove_baseline_wander,
    remove_powerline_interference,
    upsample_signals,
)
from qrs_detection import (
    FetalQRSDetectionResult,
    MaternalQRSDetectionResult,
    detect_fetal_qrs,
    detect_maternal_qrs,
)


@dataclass(frozen=True)
class PipelineParameters:
    """Parameters used by the end-to-end pipeline."""

    baseline_cutoff_hz: float = 3.0
    baseline_numtaps: int = 1001
    powerline_frequencies_hz: tuple[float, ...] = (50.0, 100.0, 150.0)
    target_fs: float = 2000.0
    maternal_max_heart_rate_bpm: float = 140.0
    mecg_preceding_beats: int = 10
    mecg_before_s: float = 0.25
    mecg_after_s: float = 0.45
    mecg_qrs_half_width_s: float = 0.05
    fetal_low_hz: float = 10.0
    fetal_high_hz: float = 60.0
    fetal_integration_window_s: float = 0.035
    fetal_threshold_percentile: float = 85.0
    fetal_min_distance_s: float = 0.25
    fetal_min_prominence: float = 0.2
    evaluation_tolerance_s: float = 0.05
    average_before_s: float = 0.2
    average_after_s: float = 0.2
    average_baseline_s: float = 0.05


@dataclass(frozen=True)
class FetalEcgPipelineResult:
    """Complete result for one record processed by the pipeline."""

    record: FetalEcgRecord
    parameters: PipelineParameters
    baseline_result: BaselineRemovalResult
    powerline_result: PowerlineRemovalResult
    upsampling_result: UpsamplingResult
    mqrs_result: MaternalQRSDetectionResult
    mecg_result: MECGCancellationResult
    fqrs_result: FetalQRSDetectionResult
    performance: QRSEvaluationResult
    reference_fqrs_s5: np.ndarray
    detected_fqrs_times_s: np.ndarray
    mean_rr_ms: float
    median_rr_ms: float
    estimated_fhr_bpm: float
    average_result: FECGAverageResult | None = None

    def to_performance_row(self) -> dict[str, float | int | str]:
        """Return a flat row suitable for a CSV performance table."""

        duration_s = self.record.signals.shape[0] / self.record.fs
        row: dict[str, float | int | str] = {
            "record": self.record.name,
            "duration_s": duration_s,
            "channels": len(self.record.channels),
            "original_fs_hz": self.record.fs,
            "pipeline_fs_hz": self.fqrs_result.fs,
            "reference_fqrs": len(self.reference_fqrs_s5),
            "detected_fqrs": len(self.fqrs_result.fqrs_samples),
            "true_positives": self.performance.true_positives,
            "false_positives": self.performance.false_positives,
            "false_negatives": self.performance.false_negatives,
            "sensitivity": self.performance.sensitivity,
            "positive_predictive_value": self.performance.positive_predictive_value,
            "f1_score": self.performance.f1_score,
            "mean_abs_error_ms": self.performance.mean_abs_error_ms,
            "tolerance_ms": self.performance.tolerance_s * 1000,
            "mqrs_detected": len(self.mqrs_result.mqrs_samples),
            "mecg_cancelled_mqrs": len(self.mecg_result.cancelled_mqrs_samples),
            "mean_rr_ms": self.mean_rr_ms,
            "median_rr_ms": self.median_rr_ms,
            "estimated_fhr_bpm": self.estimated_fhr_bpm,
        }
        if self.average_result is not None:
            row["average_used_fqrs"] = len(self.average_result.used_fqrs_samples)
            row["average_snr_gain_factor"] = self.average_result.snr_gain_factor
        return row


def _fhr_from_fqrs_times(fqrs_times_s: np.ndarray) -> tuple[float, float, float]:
    """Compute mean RR, median RR, and estimated FHR from fetal QRS times."""

    fqrs_times_s = np.asarray(fqrs_times_s, dtype=float)
    if len(fqrs_times_s) < 2:
        return np.nan, np.nan, np.nan

    rr_s = np.diff(fqrs_times_s)
    mean_rr_s = float(np.mean(rr_s))
    median_rr_s = float(np.median(rr_s))
    if mean_rr_s <= 0:
        return np.nan, median_rr_s * 1000.0, np.nan
    return mean_rr_s * 1000.0, median_rr_s * 1000.0, 60.0 / mean_rr_s


def run_pipeline_for_record(
    record_name: str,
    parameters: PipelineParameters | None = None,
    compute_average_fecg: bool = False,
) -> FetalEcgPipelineResult:
    """Run the complete fetal ECG pipeline on one PhysioNet Set-A record."""

    params = parameters or PipelineParameters()

    record = load_record(record_name)
    s1 = record.signals

    baseline_result = remove_baseline_wander(
        s1,
        fs=record.fs,
        cutoff_hz=params.baseline_cutoff_hz,
        numtaps=params.baseline_numtaps,
    )
    s2 = baseline_result.filtered_signals

    powerline_result = remove_powerline_interference(
        s2,
        fs=record.fs,
        frequencies_hz=params.powerline_frequencies_hz,
    )
    s3 = powerline_result.cleaned_signals

    upsampling_result = upsample_signals(
        s3,
        fs=record.fs,
        target_fs=params.target_fs,
    )
    s4 = upsampling_result.upsampled_signals
    pipeline_fs = upsampling_result.target_fs

    mqrs_result = detect_maternal_qrs(
        s4,
        fs=pipeline_fs,
        max_heart_rate_bpm=params.maternal_max_heart_rate_bpm,
    )

    mecg_result = cancel_maternal_ecg(
        s4,
        mqrs_samples=mqrs_result.mqrs_samples,
        fs=pipeline_fs,
        preceding_beats=params.mecg_preceding_beats,
        before_s=params.mecg_before_s,
        after_s=params.mecg_after_s,
        qrs_half_width_s=params.mecg_qrs_half_width_s,
    )
    s5 = mecg_result.cancelled_signals

    fqrs_result = detect_fetal_qrs(
        s5,
        fs=pipeline_fs,
        low_hz=params.fetal_low_hz,
        high_hz=params.fetal_high_hz,
        integration_window_s=params.fetal_integration_window_s,
        threshold_percentile=params.fetal_threshold_percentile,
        min_distance_s=params.fetal_min_distance_s,
        min_prominence=params.fetal_min_prominence,
    )

    reference_fqrs_s5 = np.round(record.fqrs_reference * pipeline_fs / record.fs).astype(int)
    performance = match_qrs_detections(
        fqrs_result.fqrs_samples,
        reference_fqrs_s5,
        fs=pipeline_fs,
        tolerance_s=params.evaluation_tolerance_s,
    )

    detected_fqrs_times_s = fqrs_result.fqrs_samples / pipeline_fs
    mean_rr_ms, median_rr_ms, estimated_fhr_bpm = _fhr_from_fqrs_times(detected_fqrs_times_s)

    average_result = None
    if compute_average_fecg:
        average_result = average_fetal_ecg(
            s5,
            fqrs_samples=fqrs_result.fqrs_samples,
            fs=pipeline_fs,
            before_s=params.average_before_s,
            after_s=params.average_after_s,
            baseline_s=params.average_baseline_s,
        )

    return FetalEcgPipelineResult(
        record=record,
        parameters=params,
        baseline_result=baseline_result,
        powerline_result=powerline_result,
        upsampling_result=upsampling_result,
        mqrs_result=mqrs_result,
        mecg_result=mecg_result,
        fqrs_result=fqrs_result,
        performance=performance,
        reference_fqrs_s5=reference_fqrs_s5,
        detected_fqrs_times_s=detected_fqrs_times_s,
        mean_rr_ms=mean_rr_ms,
        median_rr_ms=median_rr_ms,
        estimated_fhr_bpm=estimated_fhr_bpm,
        average_result=average_result,
    )
