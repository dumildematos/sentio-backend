from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# -----------------------------
# ENUMS
# -----------------------------

class EmotionType(str, Enum):
    calm = "calm"
    focused = "focused"
    relaxed = "relaxed"
    excited = "excited"
    stressed = "stressed"


class PatternType(str, Enum):
    organic = "organic"
    geometric = "geometric"
    fluid = "fluid"
    textile = "textile"


class DeviceSource(str, Enum):
    auto = "auto"
    brainflow = "brainflow"
    bluemuse = "bluemuse"


# -----------------------------
# SESSION CONFIGURATION
# -----------------------------

class SessionConfig(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str
    pattern_type: PatternType
    signal_sensitivity: float = Field(..., ge=0.0, le=1.0)
    noise_control: float = Field(..., ge=0.0, le=1.0)
    device_source: DeviceSource = Field(default=DeviceSource.auto, description="EEG transport: auto, brainflow, or bluemuse.")
    board_id: int = Field(default=38, description="BrainFlow board id. Muse 2 uses 38.")
    mac_address: Optional[str] = Field(default=None, description="Optional Bluetooth MAC address for BrainFlow.")
    serial_number: Optional[str] = Field(default=None, description="Optional BrainFlow device name or serial number for discovery.")
    serial_port: Optional[str] = Field(default=None, description="Optional serial port for BLED112 or other BrainFlow transports.")
    stream_name: Optional[str] = Field(default=None, description="Optional BlueMuse LSL stream name when multiple EEG streams are available.")
    timeout: int = Field(default=15, ge=1, le=120, description="BrainFlow connection timeout in seconds.")


class SessionStartResponse(BaseModel):
    session_id: str
    status: str


class SessionStatus(BaseModel):
    session_id: Optional[str] = None
    state: str
    start_time: Optional[float] = None
    emotion_history_length: int


# -----------------------------
# CALIBRATION
# -----------------------------

class CalibrationStatus(BaseModel):
    progress: float
    signal_quality: float
    noise_level: float
    status_message: str


# -----------------------------
# EEG DATA
# -----------------------------

class EEGData(BaseModel):
    timestamp: float
    alpha: float
    beta: float
    gamma: float
    theta: float
    delta: float
    signal_quality: float


# -----------------------------
# EMOTION DETECTION
# -----------------------------

class EmotionResult(BaseModel):
    emotion: EmotionType
    confidence: float
    mindfulness: Optional[float] = None
    restfulness: Optional[float] = None


# -----------------------------
# PATTERN PARAMETERS
# -----------------------------

class PatternParameters(BaseModel):
    pattern_type: PatternType
    color_palette: List[str]
    complexity: float
    pattern_seed: int


class StreamConfiguration(BaseModel):
    session_id: Optional[str] = None
    state: str
    device_source: DeviceSource
    board_id: int
    age: Optional[int] = None
    gender: Optional[str] = None
    sampling_rate: int
    channel_count: int
    window_size: int
    update_interval: float
    pattern_type: PatternType
    signal_sensitivity: Optional[float] = None
    noise_control: Optional[float] = None
    osc_enabled: bool
    osc_host: str
    osc_port: int
    osc_stream_address: str


# -----------------------------
# REAL-TIME STREAM MESSAGE
# -----------------------------

class BrainStreamMessage(BaseModel):
    timestamp: float

    alpha: float
    beta: float
    gamma: float
    theta: float
    delta: float

    signal_quality: float

    emotion: EmotionType
    confidence: float
    mindfulness: Optional[float] = None
    restfulness: Optional[float] = None

    pattern_seed: int
    pattern_type: PatternType
    pattern_complexity: float
    color_palette: List[str]
    config: StreamConfiguration
