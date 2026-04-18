import numpy as np
import logging
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes

try:
    from pylsl import StreamInlet, resolve_byprop
except ImportError:
    StreamInlet = None
    resolve_byprop = None


logger = logging.getLogger("sentio.muse")


class MuseConnectionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class MuseConnection:
    """
    Handles connection to a Muse board through BrainFlow or BlueMuse LSL.
    """

    def __init__(
        self,
        device_source="auto",
        board_id=38,
        mac_address=None,
        serial_number=None,
        serial_port=None,
        stream_name=None,
        timeout=15,
        stream_buffer_size=45000,
        other_info=None,
        lsl_stream_type="EEG",
        lsl_resolve_timeout=3.0,
    ):
        self.board = None
        self.inlet = None
        self.device_source = (device_source or "auto").lower()
        self.active_source = None
        self.board_id = int(board_id)
        self.sampling_rate = None
        self.eeg_channels = None
        self.mac_address = mac_address
        self.serial_number = serial_number
        self.serial_port = serial_port
        self.stream_name = stream_name
        self.timeout = timeout
        self.stream_buffer_size = int(stream_buffer_size)
        self.other_info = other_info
        self.lsl_stream_type = lsl_stream_type
        self.lsl_resolve_timeout = float(lsl_resolve_timeout)
        self.eeg_channel_indexes = []

    def _get_bluemuse_channel_indexes(self, channel_count: int) -> list[int]:
        # BlueMuse EEG streams can expose extra channels beyond the Muse EEG set.
        # Prefer the first five channels, which correspond to the Muse EEG feed.
        if channel_count <= 0:
            return []

        return list(range(min(channel_count, 5)))

    def _build_input_params(self):
        params = BrainFlowInputParams()

        if self.mac_address:
            params.mac_address = self.mac_address
        if self.serial_number:
            params.serial_number = self.serial_number
        if self.serial_port:
            params.serial_port = self.serial_port
        if self.timeout is not None:
            params.timeout = int(self.timeout)
        if self.other_info:
            params.other_info = self.other_info

        return params

    def _build_connection_error(self, error: Exception) -> BrainFlowError:
        exit_code = getattr(error, "exit_code", BrainFlowExitCodes.GENERAL_ERROR.value)
        hints = []

        if self.board_id == BoardIds.MUSE_2_BOARD.value:
            hints.append(
                "Muse 2 native BLE requires Windows 10 build 19041+ or newer and the headset paired in Windows Bluetooth settings."
            )
            if not self.mac_address and not self.serial_number:
                hints.append(
                    "If autodiscovery fails, provide mac_address or serial_number in the session payload."
                )
            hints.append(
                "Keep the headset awake and make sure BlueMuse, Muse app, or any other BLE client is not already connected unless you intentionally use device_source='bluemuse' or 'auto'."
            )
        elif self.board_id == BoardIds.MUSE_2_BLED_BOARD.value:
            if not self.serial_port:
                hints.append(
                    "Muse 2 BLED mode requires the BLED112 dongle COM port in serial_port, for example COM3."
                )
            else:
                hints.append(
                    f"Verify the BLED112 dongle is available on {self.serial_port} and not in use by another process."
                )
        elif self.board_id == BoardIds.MUSE_S_BOARD.value:
            hints.append(
                "Muse S native BLE requires Windows 10 build 19041+ or newer and the headset paired in Windows Bluetooth settings."
            )
        elif self.board_id == BoardIds.MUSE_S_BLED_BOARD.value:
            hints.append(
                "Muse S BLED mode requires the BLED112 dongle COM port in serial_port."
            )

        hint_suffix = f" {' '.join(hints)}" if hints else ""
        return BrainFlowError(f"{error}.{hint_suffix}".strip(), int(exit_code))

    def _connect_brainflow(self):
        logger.info("Preparing BrainFlow board %s", self.board_id)
        self.board = BoardShim(self.board_id, self._build_input_params())

        try:
            self.board.prepare_session()
            self.board.start_stream(self.stream_buffer_size, "")
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channel_indexes = list(BoardShim.get_eeg_channels(self.board_id))
            self.eeg_channels = len(self.eeg_channel_indexes)
            self.active_source = "brainflow"
        except Exception as exc:
            self.disconnect()
            raise self._build_connection_error(exc) from exc

        logger.info(
            "Connected to EEG stream via BrainFlow sampling_rate=%s eeg_channels=%s",
            self.sampling_rate,
            self.eeg_channels,
        )
        return True

    def _connect_bluemuse(self):
        if StreamInlet is None or resolve_byprop is None:
            raise MuseConnectionError(
                "BlueMuse mode requires pylsl to be installed and available in the backend environment.",
                status_code=500,
            )

        logger.info(
            "Resolving BlueMuse LSL EEG stream type=%s stream_name=%s",
            self.lsl_stream_type,
            self.stream_name,
        )
        streams = resolve_byprop("type", self.lsl_stream_type, timeout=self.lsl_resolve_timeout)

        if self.stream_name:
            streams = [stream for stream in streams if stream.name() == self.stream_name]

        if not streams:
            stream_hint = f" named '{self.stream_name}'" if self.stream_name else ""
            raise MuseConnectionError(
                f"No BlueMuse LSL EEG streams found{stream_hint}. Start BlueMuse, enable its EEG LSL stream, and keep BlueMuse connected to the headset.",
                status_code=400,
            )

        self.inlet = StreamInlet(streams[0])
        stream_info = self.inlet.info()
        nominal_rate = stream_info.nominal_srate()
        self.sampling_rate = int(nominal_rate) if nominal_rate else 256
        self.eeg_channels = int(stream_info.channel_count())
        self.eeg_channel_indexes = self._get_bluemuse_channel_indexes(self.eeg_channels)
        self.active_source = "bluemuse"
        logger.info(
            "Connected to EEG stream via BlueMuse LSL sampling_rate=%s eeg_channels=%s selected_indexes=%s",
            self.sampling_rate,
            self.eeg_channels,
            self.eeg_channel_indexes,
        )
        return True

    def _connect_auto(self):
        bluemuse_error = None

        try:
            return self._connect_bluemuse()
        except MuseConnectionError as exc:
            bluemuse_error = exc

        try:
            return self._connect_brainflow()
        except BrainFlowError as exc:
            raise MuseConnectionError(
                f"BlueMuse LSL connection failed: {bluemuse_error} BrainFlow fallback failed: {exc}. If you use BlueMuse, enable its EEG LSL stream and keep device_source as 'auto' or set it to 'bluemuse'. If you want direct BrainFlow BLE, close BlueMuse first.",
                status_code=400,
            ) from exc

    def is_connected(self):
        if self.active_source == "bluemuse":
            return self.inlet is not None

        return self.board is not None and self.board.is_prepared()

    def connect(self):
        """
        Connect to a BrainFlow board or BlueMuse LSL stream.
        """
        self.disconnect()

        if self.device_source == "brainflow":
            return self._connect_brainflow()

        if self.device_source == "bluemuse":
            return self._connect_bluemuse()

        if self.device_source == "auto":
            return self._connect_auto()

        raise MuseConnectionError(
            f"Unsupported device_source '{self.device_source}'. Use 'auto', 'brainflow', or 'bluemuse'.",
            status_code=400,
        )

    def get_eeg_data(self, window_size=256):
        """
        Retrieve the latest EEG window as samples x channels.
        """
        if self.active_source == "bluemuse":
            if self.inlet is None:
                return None

            samples, _ = self.inlet.pull_chunk(timeout=0.25, max_samples=int(window_size))
            if not samples or len(samples) < 64:
                return None

            sample_array = np.asarray(samples, dtype=np.float64)
            if sample_array.ndim == 1:
                sample_array = sample_array.reshape(-1, 1)

            if sample_array.ndim != 2:
                return None

            if self.eeg_channel_indexes:
                valid_indexes = [
                    index for index in self.eeg_channel_indexes if index < sample_array.shape[1]
                ]
                if not valid_indexes:
                    return None
                sample_array = sample_array[:, valid_indexes]

            finite_row_mask = np.all(np.isfinite(sample_array), axis=1)
            sample_array = sample_array[finite_row_mask]
            if sample_array.shape[0] < 64 or sample_array.shape[1] == 0:
                return None

            return sample_array

        if not self.is_connected():
            return None

        available_samples = self.board.get_board_data_count()
        if available_samples <= 0:
            return None

        sample_count = min(int(window_size), int(available_samples))
        if sample_count < 64:
            return None

        board_data = self.board.get_current_board_data(sample_count)
        if board_data.size == 0 or not self.eeg_channel_indexes:
            return None

        eeg_data = board_data[self.eeg_channel_indexes, :]
        if eeg_data.size == 0:
            return None

        return np.asarray(eeg_data.T, dtype=np.float64)

    def get_runtime_diagnostics(self) -> dict:
        diagnostics = {
            "device_source": self.device_source,
            "active_source": self.active_source,
            "sampling_rate": self.sampling_rate,
            "eeg_channels": self.eeg_channels,
            "eeg_channel_indexes": self.eeg_channel_indexes,
        }

        if self.active_source == "bluemuse":
            diagnostics["inlet_ready"] = self.inlet is not None
            return diagnostics

        board = self.board
        diagnostics["board_ready"] = board is not None

        if board is not None:
            try:
                diagnostics["board_prepared"] = bool(board.is_prepared())
            except Exception as exc:
                diagnostics["board_prepared_error"] = str(exc)

            try:
                diagnostics["available_samples"] = int(board.get_board_data_count())
            except Exception as exc:
                diagnostics["available_samples_error"] = str(exc)

        return diagnostics

    def disconnect(self):
        """
        Stop BrainFlow streaming and release the session.
        """
        had_session = self.board is not None or self.inlet is not None

        if self.board is not None:
            try:
                if self.board.is_prepared():
                    try:
                        self.board.stop_stream()
                    except Exception:
                        pass
                    self.board.release_session()
            finally:
                self.board = None

        self.inlet = None
        self.active_source = None

        if had_session:
            logger.info("EEG stream disconnected")