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


# -----------------------------
# SESSION CONFIGURATION
# -----------------------------

class SessionConfig(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: str
    pattern_type: PatternType
    signal_sensitivity: float = Field(..., ge=0.0, le=1.0)
    noise_control: float = Field(..., ge=0.0, le=1.0)
    mac_address: str = Field(..., description="Muse 2 device MAC address")


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

    pattern_seed: int
    pattern_type: PatternType
    pattern_complexity: float
    color_palette: List[str]
    config: StreamConfiguration
