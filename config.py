from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global configuration for Sentio backend.
    """

    # FastAPI / server
    app_name: str = "Sentio EEG Backend"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost",
            "http://127.0.0.1",
            "http://10.208.193.106",
        ]
    )
    cors_allowed_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1|10\.208\.193\.106)(:\d+)?$"

    # BrainFlow / Muse 2 configuration
    muse_device_source: str = "auto"
    muse_board_id: int = 38  # BoardIds.MUSE_2_BOARD.value
    muse_sampling_rate: int = 256  # Hz
    muse_window_size: int = 256  # samples per read
    muse_mac_address: str | None = None
    muse_serial_number: str | None = None
    brainflow_connection_timeout: int = 15
    brainflow_stream_buffer_size: int = 45000
    bluemuse_stream_name: str | None = None
    bluemuse_lsl_stream_type: str = "EEG"
    bluemuse_lsl_resolve_timeout: float = 3.0

    # EEG processing
    eeg_update_interval: float = 0.2  # seconds between WebSocket updates

    # Calibration
    calibration_duration: int = 5  # seconds for baseline
    noise_threshold: float = 0.5  # optional: signal quality threshold

    # Pattern mapping
    default_pattern_type: str = "organic"

    # WebSocket
    ws_endpoint: str = "/ws/brain-stream"

    # OSC / TouchDesigner
    osc_enabled: bool = True
    osc_host: str = "127.0.0.1"
    osc_port: int = 7000
    osc_stream_address: str = "/sentio"


# Single global settings instance
settings = Settings()