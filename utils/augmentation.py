"""Utility functions for ECG data augmentation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import neurokit2 as nk
import numpy as np


@dataclass
class GaussianNoiseConfig:
    """Configuration for Gaussian noise augmentation.

    Attributes
    ----------
    std: float
        Noise standard deviation. If ``relative`` is ``True`` this is interpreted
        as a multiple of the sample's standard deviation.
    relative: bool
        Whether ``std`` should scale with the sample's standard deviation.
    random_state: Optional[int]
        Seed for the random number generator used to sample noise.
    """

    std: float = 0.01
    relative: bool = True
    random_state: Optional[int] = None


def add_gaussian_noise(sample: np.ndarray, config: GaussianNoiseConfig | None = None) -> np.ndarray:
    """Additive Gaussian noise augmentation for a single ECG sample.

    Parameters
    ----------
    sample:
        One-dimensional ECG sample.
    config:
        Optional :class:`GaussianNoiseConfig` instance controlling the noise
        properties. By default noise with a standard deviation equal to 1% of the
        signal standard deviation is used.

    Returns
    -------
    numpy.ndarray
        Augmented sample with additive Gaussian noise.
    """

    if config is None:
        config = GaussianNoiseConfig()

    rng = np.random.default_rng(seed=config.random_state)
    scale = config.std * np.std(sample) if config.relative else config.std
    if scale <= 0:
        return sample.copy()

    noise = rng.normal(loc=0.0, scale=scale, size=sample.shape)
    return sample + noise


def _last_valid_index(values: Iterable[float]) -> Optional[int]:
    last_valid = None
    for value in values:
        if not np.isnan(value):
            last_valid = int(value)
    return last_valid


def remove_post_pqrst_segment(
    sample: np.ndarray,
    sampling_rate: int,
    delineate_method: str = "dwt",
    peak_detection_method: str = "neurokit",
    pad_ms: float = 40.0,
) -> np.ndarray:
    """Zero out the portion of the ECG sample after the final T-wave offset.

    The function keeps the original signal length while removing the tail that
    typically contains baseline drift or non-PQRST activity.

    Parameters
    ----------
    sample:
        One-dimensional ECG sample.
    sampling_rate:
        Sampling frequency of ``sample`` in Hertz.
    delineate_method:
        Method passed to :func:`neurokit2.ecg_delineate` (``"dwt"`` by default).
    peak_detection_method:
        Method used in :func:`neurokit2.ecg_peaks` to detect R-peaks.
    pad_ms:
        Extra duration (in milliseconds) to keep after the detected T-wave
        offset before zeroing the remainder of the sample.

    Returns
    -------
    numpy.ndarray
        Sample with the non-PQRST tail replaced by zeros. If delineation fails
        the original sample is returned.
    """

    if len(sample) == 0:
        return sample.copy()

    try:
        _, rpeaks_info = nk.ecg_peaks(sample, sampling_rate=sampling_rate, method=peak_detection_method)
        r_locs = rpeaks_info.get("ECG_R_Peaks", None)
        if r_locs is None or len(r_locs) == 0:
            return sample.copy()

        _, delineate_info = nk.ecg_delineate(
            sample,
            rpeaks=rpeaks_info,
            sampling_rate=sampling_rate,
            method=delineate_method,
        )
        t_offsets = delineate_info.get("ECG_T_Offsets", None)
        if t_offsets is None:
            return sample.copy()

        last_t_offset = _last_valid_index(t_offsets)
        if last_t_offset is None:
            return sample.copy()

        pad_samples = int(round((pad_ms / 1000.0) * sampling_rate))
        cutoff_index = min(len(sample), last_t_offset + pad_samples)
    except Exception:  # pragma: no cover - fall back to original when delineation fails
        return sample.copy()

    trimmed = sample.copy()
    trimmed[cutoff_index:] = 0.0
    return trimmed


def augment_sample(
    sample: np.ndarray,
    sampling_rate: int,
    noise_config: GaussianNoiseConfig | None = None,
    delineate_method: str = "dwt",
    peak_detection_method: str = "neurokit",
    pad_ms: float = 40.0,
) -> np.ndarray:
    """Apply Gaussian noise and tail removal augmentation to an ECG sample."""

    trimmed = remove_post_pqrst_segment(
        sample,
        sampling_rate=sampling_rate,
        delineate_method=delineate_method,
        peak_detection_method=peak_detection_method,
        pad_ms=pad_ms,
    )

    return add_gaussian_noise(trimmed, config=noise_config)


def augment_batch(
    samples: np.ndarray,
    sampling_rate: int,
    noise_config: GaussianNoiseConfig | None = None,
    delineate_method: str = "dwt",
    peak_detection_method: str = "neurokit",
    pad_ms: float = 40.0,
) -> np.ndarray:
    """Vectorised augmentation for a batch of ECG samples."""

    augmented_samples = np.empty_like(samples)
    for idx, sample in enumerate(samples):
        augmented_samples[idx] = augment_sample(
            sample,
            sampling_rate=sampling_rate,
            noise_config=noise_config,
            delineate_method=delineate_method,
            peak_detection_method=peak_detection_method,
            pad_ms=pad_ms,
        )
    return augmented_samples


__all__ = [
    "GaussianNoiseConfig",
    "add_gaussian_noise",
    "remove_post_pqrst_segment",
    "augment_sample",
    "augment_batch",
]
