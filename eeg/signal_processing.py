import numpy as np
from scipy.signal import welch


class SignalProcessor:
    """
    Extract EEG frequency band powers from raw signals.
    """

    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate

    def bandpower(self, data, band):
        """
        Compute power of a specific frequency band using Welch PSD.
        """
        if data is None or len(data) < 2:
            return 0.0

        low, high = band
        segment_length = min(self.sampling_rate, len(data))

        freqs, psd = welch(
            data,
            fs=self.sampling_rate,
            nperseg=segment_length
        )

        band_mask = (freqs >= low) & (freqs <= high)
        if not np.any(band_mask):
            return 0.0

        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])

        return band_power

    def extract_features(self, eeg_data):
        """
        Compute EEG band powers.
        """

        if eeg_data is None or len(eeg_data) == 0:
            return None

        eeg_array = np.asarray(eeg_data)
        if eeg_array.ndim == 1:
            signal = eeg_array
        else:
            signal = np.mean(eeg_array, axis=1)

        alpha = self.bandpower(signal, (8, 12))
        beta = self.bandpower(signal, (13, 30))
        gamma = self.bandpower(signal, (30, 45))
        theta = self.bandpower(signal, (4, 7))
        delta = self.bandpower(signal, (1, 4))

        total = alpha + beta + gamma + theta + delta + 1e-6

        return {
            "alpha": alpha / total,
            "beta": beta / total,
            "gamma": gamma / total,
            "theta": theta / total,
            "delta": delta / total
        }