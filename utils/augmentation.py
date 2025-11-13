"""Utility functions for ECG data augmentation tailored to the MIT-BIH dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


MITBIH_SAMPLING_RATE = 360
"""Nominal sampling rate (Hz) used for the MIT-BIH heartbeat segments."""

MITBIH_SAMPLE_LENGTH = 187
"""Number of samples per heartbeat segment in the Kaggle MIT-BIH dataset."""


@dataclass
class GaussianNoiseConfig:
    """Configuration for Gaussian noise augmentation of MIT-BIH beats."""

    std: float = 0.015
    relative: bool = True
    random_state: Optional[int] = None


@dataclass
class TailTrimmingConfig:
    """Configuration for trimming the non-PQRST tail of MIT-BIH beats."""

    amplitude_ratio: float = 0.08
    """Minimum relative amplitude to keep (fraction of the smoothed max)."""

    smoothing_window: int = 5
    """Window size (in samples) for smoothing the absolute amplitude envelope."""

    pad: int = 6
    """Number of samples to retain after the last significant activity index."""


def _as_float_array(sample: np.ndarray) -> np.ndarray:
    array = np.asarray(sample, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("MIT-BIH heartbeat samples must be one-dimensional.")
    if array.size != MITBIH_SAMPLE_LENGTH:
        raise ValueError(
            f"Expected MIT-BIH heartbeat length {MITBIH_SAMPLE_LENGTH}, received {array.size}."
        )
    return array


def add_gaussian_noise(sample: np.ndarray, config: GaussianNoiseConfig | None = None) -> np.ndarray:
    """Inject Gaussian noise into a MIT-BIH heartbeat sample."""

    sample_arr = _as_float_array(sample)
    if config is None:
        config = GaussianNoiseConfig()

    rng = np.random.default_rng(seed=config.random_state)
    scale = config.std * np.std(sample_arr) if config.relative else config.std
    if not np.isfinite(scale) or scale <= 0:
        return sample_arr.copy()

    noise = rng.normal(loc=0.0, scale=scale, size=sample_arr.shape)
    return sample_arr + noise.astype(sample_arr.dtype, copy=False)


def _resample_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a one-dimensional signal to a target length via linear interpolation."""

    if target_length <= 0:
        raise ValueError("target_length must be positive")

    if signal.size == target_length:
        return signal.astype(np.float32, copy=True)

    if signal.size < 2:
        # Nothing to interpolate, pad or truncate directly.
        result = np.zeros(target_length, dtype=np.float32)
        result[: min(target_length, signal.size)] = signal.astype(np.float32, copy=False)
        return result

    xp = np.linspace(0.0, 1.0, num=signal.size, endpoint=True, dtype=np.float32)
    fp = signal.astype(np.float32, copy=False)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=True, dtype=np.float32)
    return np.interp(x_new, xp, fp).astype(np.float32, copy=False)


def remove_mitbih_tail(sample: np.ndarray, config: TailTrimmingConfig | None = None) -> np.ndarray:
    """Remove the low-activity tail and rescale to the original MIT-BIH span."""

    sample_arr = _as_float_array(sample)
    if config is None:
        config = TailTrimmingConfig()

    if sample_arr.size == 0:
        return sample_arr.copy()

    window = max(1, int(config.smoothing_window))
    if window > sample_arr.size:
        window = sample_arr.size

    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(np.abs(sample_arr), kernel, mode="same")
    peak = float(smoothed.max())
    if peak <= 0:
        return sample_arr.copy()

    threshold = peak * float(config.amplitude_ratio)
    if threshold <= 0:
        return sample_arr.copy()

    significant = np.flatnonzero(smoothed >= threshold)
    if significant.size == 0:
        return sample_arr.copy()

    last_idx = significant[-1] + int(config.pad)
    cutoff = min(sample_arr.size, max(last_idx, significant[-1] + 1))

    trimmed_segment = sample_arr[:cutoff]
    return _resample_to_length(trimmed_segment, MITBIH_SAMPLE_LENGTH)


def augment_sample(
    sample: np.ndarray,
    noise_config: GaussianNoiseConfig | None = None,
    tail_config: TailTrimmingConfig | None = None,
) -> np.ndarray:
    """Apply MIT-BIH specific augmentations to a single heartbeat segment."""

    trimmed = remove_mitbih_tail(sample, config=tail_config)
    return add_gaussian_noise(trimmed, config=noise_config)


def augment_batch(
    samples: np.ndarray,
    noise_config: GaussianNoiseConfig | None = None,
    tail_config: TailTrimmingConfig | None = None,
) -> np.ndarray:
    """Vectorised augmentation for a batch of MIT-BIH heartbeat segments."""

    samples_arr = np.asarray(samples, dtype=np.float32)
    if samples_arr.ndim != 2:
        raise ValueError("MIT-BIH heartbeat batches must be two-dimensional.")
    if samples_arr.shape[1] != MITBIH_SAMPLE_LENGTH:
        raise ValueError(
            f"Expected MIT-BIH heartbeat length {MITBIH_SAMPLE_LENGTH}, received {samples_arr.shape[1]}."
        )

    augmented = np.empty_like(samples_arr)
    for idx, sample in enumerate(samples_arr):
        augmented[idx] = augment_sample(sample, noise_config=noise_config, tail_config=tail_config)
    return augmented


__all__ = [
    "MITBIH_SAMPLE_LENGTH",
    "MITBIH_SAMPLING_RATE",
    "GaussianNoiseConfig",
    "TailTrimmingConfig",
    "add_gaussian_noise",
    "remove_mitbih_tail",
    "augment_sample",
    "augment_batch",
]
