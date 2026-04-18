from fastapi import APIRouter, HTTPException
from brainflow.exit_codes import BrainFlowError, BrainFlowExitCodes
from config import settings
from eeg.muse_connection import MuseConnection, MuseConnectionError
from services.session_manager import SessionState, session_manager
from eeg.calibration import CalibrationManager
from patterns.pattern_mapper import PatternMapper
from services.stream_service import osc_sender, start_streaming
from models.schemas import (
    SessionConfig,
    SessionStartResponse,
    SessionStatus,
    CalibrationStatus,
    EmotionType,
    PatternParameters,
    PatternType
)

router = APIRouter()


# Initialize calibration and pattern modules
calibration_manager = CalibrationManager()
pattern_mapper = PatternMapper()


def _build_muse_connection(session_config: dict) -> MuseConnection:
    return MuseConnection(
        device_source=session_config.get("device_source") or settings.muse_device_source,
        board_id=int(session_config.get("board_id") or settings.muse_board_id),
        mac_address=session_config.get("mac_address") or settings.muse_mac_address,
        serial_number=session_config.get("serial_number") or settings.muse_serial_number,
        serial_port=session_config.get("serial_port"),
        stream_name=session_config.get("stream_name") or settings.bluemuse_stream_name,
        timeout=int(session_config.get("timeout") or settings.brainflow_connection_timeout),
        stream_buffer_size=settings.brainflow_stream_buffer_size,
        lsl_stream_type=settings.bluemuse_lsl_stream_type,
        lsl_resolve_timeout=settings.bluemuse_lsl_resolve_timeout,
    )


def _brainflow_http_status(error: BrainFlowError) -> int:
    recoverable_codes = {
        BrainFlowExitCodes.BOARD_NOT_READY_ERROR.value,
        BrainFlowExitCodes.INVALID_ARGUMENTS_ERROR.value,
        BrainFlowExitCodes.SYNC_TIMEOUT_ERROR.value,
        BrainFlowExitCodes.PORT_ALREADY_OPEN_ERROR.value,
        BrainFlowExitCodes.UNABLE_TO_OPEN_PORT_ERROR.value,
        BrainFlowExitCodes.SER_PORT_ERROR.value,
    }
    return 400 if error.exit_code in recoverable_codes else 500


# Muse device connection will be created per session


# -----------------------------
# SESSION ENDPOINTS
# -----------------------------


@router.post(
    "/session/start",
    response_model=SessionStartResponse,
    responses={400: {"description": "Device connection problem"}, 500: {"description": "Unexpected startup failure"}},
)
def start_session(config: SessionConfig):
    """
    Start a new EEG session and check device connection.
    """
    session_config = config.model_dump(mode="json")

    # Try to connect to Muse device
    try:
        muse_connection = _build_muse_connection(session_config)
        muse_connection.connect()
    except MuseConnectionError as e:
        raise HTTPException(status_code=e.status_code, detail=f"Device connection failed: {str(e)}")
    except BrainFlowError as e:
        raise HTTPException(status_code=_brainflow_http_status(e), detail=f"Device connection failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Device connection failed: {str(e)}")

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

@router.get(
    "/calibration/run",
    response_model=CalibrationStatus,
    responses={400: {"description": "No active session"}, 500: {"description": "Calibration failure"}},
)
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
        raise HTTPException(status_code=500, detail="No BrainFlow connection instance found for calibration")
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

@router.get(
    "/pattern/generate",
    response_model=PatternParameters,
    responses={400: {"description": "Invalid emotion supplied"}},
)
def generate_pattern(emotion: str, pattern_type: PatternType):
    """
    Generate pattern parameters based on emotion and EEG features.
    """
    # Use the last emotion from session history if available
    if session_manager.emotion_history:
        latest_emotion = session_manager.emotion_history[-1].get("emotion", emotion)
    else:
        latest_emotion = emotion

    try:
        emotion_value = EmotionType(latest_emotion)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown emotion: {latest_emotion}") from exc

    latest_stream_message = session_manager.get_latest_stream_message() or {}
    eeg_features = {
        band: float(latest_stream_message.get(band, 0.0) or 0.0)
        for band in ("alpha", "beta", "gamma", "theta", "delta")
    }


    pattern_params = pattern_mapper.map_pattern(
        emotion=emotion_value,
        eeg_features=eeg_features,
        selected_pattern=pattern_type
    )

    return pattern_params