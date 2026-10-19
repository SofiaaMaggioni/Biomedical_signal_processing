# Biomedical_signal_processing
Biomedical_signal_processing_project 

Grade: 24

AY: 2025-2026

# Non-Invasive Fetal ECG Extraction & Detection

This repository provides a modular, end-to-end signal processing pipeline for the non-invasive extraction and robust detection of the fetal electrocardiogram (**fECG**) from multichannel maternal abdominal recordings.

The algorithmic workflow reproduces and builds upon the method proposed by **Martens et al. (2007)** (*"A robust fetal ECG detection method for abdominal recordings"*, *Physiological Measurement*), with quantitative benchmark validation against clinical ground-truth annotations from the **PhysioNet / Computing in Cardiology Challenge 2013 (Set-A)**.

### Pipeline Architecture
The project is organized into structured, reproducible modules:
**Data Ingestion (S1):** Automated loading of 4-channel abdominal recordings (1000 Hz, 60 s) and reference annotations (.fqrs) via `wfdb`.
**Pre-processing (S1 -> S4):**
  - Zero-phase high-pass FIR filtering (f_c = 3.0\Hz, 1001 taps) for baseline wander removal without group delay distortion (S2).
  - Power-line interference cancellation (50 Hz and harmonics at 100/150 Hz) (S3).
  - Temporal upsampling to 2000 Hz (S4) to achieve sub-millisecond alignment resolution.
**Maternal ECG Cancellation (S4 -> S5):**
  - Spatial PCA decomposition and matched filtering (cross-correlation) for robust mQRS localization.
  - Beat-by-beat adaptive maternal template subtraction using a 10-beat moving window and independent 3-zone scaling (P-wave, QRS, T-wave).
**Fetal QRS Detection & Benchmarking:**
  - Band-pass energy integration (10–60 Hz), moving window integration (35 ms), and dynamic percentile thresholding with a physiological refractory period (250 ms).
  - Quantitative validation (Sensitivity, PPV, F1-score) against clinical reference annotations within the standard CinC 2013 tolerance window (50ms).
**Morphological Averaging & Batch Testing (s6):**
  - Coherent ensemble averaging of detected fetal complexes for S6 average beat reconstruction (theoretical SNR gain of sqrt).
  - Cross-record batch evaluation across multiple Set-A recordings with fixed parameters to test algorithmic robustness and edge cases.
