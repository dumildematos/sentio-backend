import time
import numpy as np


class CalibrationManager:
    """
    Runs calibration phase to measure baseline EEG and signal quality.
    """

    def __init__(self, duration=5):
        self.duration = duration
        self.baseline = None
        self.signal_quality = 0

    def run_calibration(self, muse_connection):
        """
        Collect baseline EEG samples.
        """
        samples = self._collect_samples(muse_connection)

        if len(samples) == 0:
            return None

        normalized_chunks = self._normalize_chunks(samples)

        if not normalized_chunks:
            return None

        samples = np.concatenate(normalized_chunks, axis=0)

        self.baseline = np.mean(samples)
        self.signal_quality = self._compute_signal_quality(samples)

        return {
            "baseline": float(self.baseline),
            "signal_quality": float(self.signal_quality)
        }

    def _collect_samples(self, muse_connection):
        start_time = time.time()
        samples = []

        while time.time() - start_time < self.duration:
            eeg = muse_connection.get_eeg_data()
            if eeg is not None:
                samples.append(eeg)
            time.sleep(0.1)

        return samples

    def _normalize_chunks(self, chunks):
        normalized_chunks = []
        expected_channel_count = None

        for eeg_chunk in chunks:
            chunk = np.asarray(eeg_chunk)
            if chunk.size == 0:
                continue

            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            if chunk.ndim != 2:
                continue

            if expected_channel_count is None:
                expected_channel_count = chunk.shape[1]

            if chunk.shape[1] != expected_channel_count:
                continue

            normalized_chunks.append(chunk)

        return normalized_chunks

    def _compute_signal_quality(self, data):
        """
        Simple signal quality estimation based on variance.
        """

        variance = np.var(data)

        if variance == 0:
            return 0

        quality = min(variance / 1000, 1.0)

        return quality