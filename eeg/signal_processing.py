import numpy as np
import logging
from brainflow.data_filter import DataFilter


logger = logging.getLogger("sentio.signal")


class SignalProcessor:
    """
    Extract EEG frequency band powers from raw signals using BrainFlow.
    """

    def __init__(self, sampling_rate=256):
        self.sampling_rate = int(sampling_rate)

    def _normalize_channel_major(self, eeg_data):
        eeg_array = np.asarray(eeg_data, dtype=np.float64)
        if eeg_array.ndim == 1:
            eeg_array = eeg_array.reshape(-1, 1)

        if eeg_array.ndim != 2:
            return None

        channel_major = eeg_array.T if eeg_array.shape[0] >= eeg_array.shape[1] else eeg_array
        if channel_major.shape[0] == 0 or channel_major.shape[1] < 64:
            return None

        finite_channel_mask = np.all(np.isfinite(channel_major), axis=1)
        channel_major = channel_major[finite_channel_mask]
        if channel_major.shape[0] == 0 or channel_major.shape[1] < 64:
            return None

        return np.ascontiguousarray(channel_major, dtype=np.float64)

    def extract_features(self, eeg_data):
        """
        Compute EEG band powers and the BrainFlow feature vector.
        """

        if eeg_data is None or len(eeg_data) == 0:
            return None

        channel_major = self._normalize_channel_major(eeg_data)
        if channel_major is None:
            return None

        try:
            avg_band_powers, std_band_powers = DataFilter.get_avg_band_powers(
                channel_major,
                list(range(channel_major.shape[0])),
                self.sampling_rate,
                True,
            )
        except Exception as exc:
            logger.warning(
                "Band power extraction failed for shape=%s sampling_rate=%s: %s",
                channel_major.shape,
                self.sampling_rate,
                exc,
            )
            return None

        total = float(np.sum(avg_band_powers)) + 1e-6
        feature_vector = np.concatenate((avg_band_powers, std_band_powers)).astype(float)

        return {
            "alpha": float(avg_band_powers[2] / total),
            "beta": float(avg_band_powers[3] / total),
            "gamma": float(avg_band_powers[4] / total),
            "theta": float(avg_band_powers[1] / total),
            "delta": float(avg_band_powers[0] / total),
            "brainflow_feature_vector": feature_vector.tolist(),
        }