import numpy as np
from pylsl import StreamInlet, resolve_byprop


class MuseConnection:
    """
    Handles connection to a BlueMuse EEG LSL stream.
    """

    def __init__(self, mac_address=None):
        self.inlet = None
        self.sampling_rate = None
        self.eeg_channels = None
        self.mac_address = mac_address

    def connect(self):
        """
        Connect to EEG stream from BlueMuse using pylsl.
        """
        print("Resolving EEG streams...")
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if not streams:
            print("No EEG streams found.")
            return False
        self.inlet = StreamInlet(streams[0])
        self.sampling_rate = self.inlet.info().nominal_srate()
        self.eeg_channels = self.inlet.info().channel_count()
        print("Connected to EEG stream via BlueMuse.")
        return True

    def get_eeg_data(self, window_size=256):
        """
        Retrieve latest EEG samples from LSL stream.
        """
        if not self.inlet:
            return None
        samples, _ = self.inlet.pull_chunk(timeout=0.25, max_samples=window_size)
        if not samples:
            return None
        return np.asarray(samples)

    def disconnect(self):
        """
        Release LSL inlet.
        """
        self.inlet = None
        print("EEG stream disconnected.")