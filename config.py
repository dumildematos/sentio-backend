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

    # Muse 2 configuration
    muse_board_id: int = 22  # BoardIds.MUSE_2_BOARD.value
    muse_sampling_rate: int = 256  # Hz
    muse_window_size: int = 256  # samples per read
    muse_mac_address: str = "00:11:22:33:44:55"  # Replace with your Muse 2 MAC address

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