from fastapi import APIRouter, HTTPException
from eeg.muse_connection import MuseConnection
from services.session_manager import SessionState, session_manager
from eeg.calibration import CalibrationManager
from patterns.pattern_mapper import PatternMapper
from services.stream_service import osc_sender, start_streaming
from models.schemas import (
    SessionConfig,
    SessionStartResponse,
    SessionStatus,
    CalibrationStatus,
    PatternParameters,
    PatternType
)

router = APIRouter()


# Initialize calibration and pattern modules
calibration_manager = CalibrationManager()
pattern_mapper = PatternMapper()

from config import settings
 # Muse device connection will be created per session


# -----------------------------
# SESSION ENDPOINTS
# -----------------------------


@router.post("/session/start", response_model=SessionStartResponse)
def start_session(config: SessionConfig):
    """
    Start a new EEG session and check device connection.
    """
    # Try to connect to Muse device
    try:
        muse_connection = MuseConnection(mac_address=config.mac_address)
        connected = muse_connection.connect()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Device connection failed: {str(e)}")
    if not connected:
        raise HTTPException(status_code=400, detail="Muse device not connected")

    session_config = config.model_dump(mode="json")
    session_id = session_manager.start_session(session_config)
    session_manager.muse_connection = muse_connection
    session_manager.set_state(SessionState.CONNECTING)
    osc_sender.send_fields(
        {
            "age": session_config.get("age"),
            "pattern_type": session_config.get("pattern_type"),
        }
    )
    start_streaming()
    return {"session_id": session_id, "status": "started"}


@router.post("/session/stop")
def stop_session():
    """
    Stop the active session.
    """
    session_manager.stop_session()
    return {"status": "stopped"}


@router.get("/session/status", response_model=SessionStatus)
def get_session_status():
    """
    Return current session info.
    """
    info = session_manager.get_session_info()
    return SessionStatus(
        session_id=info["session_id"],
        state=info["state"],
        start_time=info["start_time"],
        emotion_history_length=info["emotion_history_length"]
    )


# -----------------------------
# CALIBRATION ENDPOINT
# -----------------------------

@router.get("/calibration/run", response_model=CalibrationStatus)
def run_calibration():
    """
    Run EEG calibration (baseline + signal quality).
    """
    if not session_manager.is_active():
        raise HTTPException(status_code=400, detail="No active session")

    # Here we would pass the Muse connection to calibration
    # Retrieve the active MuseConnection from session_manager
    muse_connection = getattr(session_manager, 'muse_connection', None)
    if muse_connection is None:
        raise HTTPException(status_code=500, detail="No MuseConnection instance found for calibration")
    was_streaming = session_manager.is_streaming()

    if was_streaming:
        session_manager.set_state(SessionState.CALIBRATING)
        session_manager.request_stream_stop()
        session_manager.wait_for_stream_stop(timeout=2.0)

    try:
        calibration_result = calibration_manager.run_calibration(
            muse_connection=muse_connection
        )

        if calibration_result is None:
            raise HTTPException(status_code=500, detail="Calibration failed")

        return CalibrationStatus(
            progress=1.0,  # Simplified for demo
            signal_quality=calibration_result["signal_quality"],
            noise_level=0.0,  # Optional: compute from EEG data
            status_message="Calibration complete"
        )
    finally:
        if was_streaming and session_manager.current_session_id is not None:
            session_manager.set_state(SessionState.CONNECTING)
            start_streaming()


# -----------------------------
# PATTERN ENDPOINTS
# -----------------------------

@router.get("/pattern/generate", response_model=PatternParameters)
def generate_pattern(emotion: str, pattern_type: PatternType):
    """
    Generate pattern parameters based on emotion and EEG features.
    """
    # Use the last emotion from session history if available
    if session_manager.emotion_history:
        emotion_value = session_manager.emotion_history[-1]
    else:
        emotion_value = emotion


        # Try to connect to Muse device using MAC address from payload
        muse_connection = MuseConnection(mac_address=config.mac_address)
        try:
            connected = muse_connection.connect()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Device connection failed: {str(e)}")
        if not connected:
            raise HTTPException(status_code=400, detail="Muse device not connected")


    pattern_params = pattern_mapper.map_pattern(
        emotion=emotion_value,
        eeg_features=dummy_features,
        selected_pattern=pattern_type
    )

    return pattern_params