import time
import logging

from config import settings
from eeg.muse_connection import MuseConnection
from eeg.signal_processing import SignalProcessor
from emotion.emotion_model import EmotionModel
from models.schemas import PatternType
from patterns.pattern_mapper import PatternMapper
from services.osc_sender import OscStreamSender
from services.session_manager import SessionState, session_manager

logger = logging.getLogger("sentio.stream")

processor = SignalProcessor(settings.muse_sampling_rate)
emotion_model = EmotionModel()
pattern_mapper = PatternMapper()
osc_sender = OscStreamSender(
    enabled=settings.osc_enabled,
    host=settings.osc_host,
    port=settings.osc_port,
    stream_address=settings.osc_stream_address,
)


def _get_selected_pattern() -> PatternType:
    configured_pattern = session_manager.session_config.get(
        "pattern_type",
        settings.default_pattern_type,
    )
    if isinstance(configured_pattern, PatternType):
        return configured_pattern

    try:
        return PatternType(configured_pattern)
    except ValueError:
        return PatternType(settings.default_pattern_type)


def _get_or_create_muse_connection() -> MuseConnection:
    muse_connection = session_manager.muse_connection
    if muse_connection is None:
        muse_connection = MuseConnection(
            device_source=session_manager.session_config.get("device_source") or settings.muse_device_source,
            board_id=int(session_manager.session_config.get("board_id") or settings.muse_board_id),
            mac_address=session_manager.session_config.get("mac_address") or settings.muse_mac_address,
            serial_number=session_manager.session_config.get("serial_number") or settings.muse_serial_number,
            serial_port=session_manager.session_config.get("serial_port"),
            stream_name=session_manager.session_config.get("stream_name") or settings.bluemuse_stream_name,
            timeout=int(
                session_manager.session_config.get("timeout") or settings.brainflow_connection_timeout
            ),
            stream_buffer_size=settings.brainflow_stream_buffer_size,
            lsl_stream_type=settings.bluemuse_lsl_stream_type,
            lsl_resolve_timeout=settings.bluemuse_lsl_resolve_timeout,
        )
        session_manager.muse_connection = muse_connection
    return muse_connection


def _build_stream_config(muse_connection: MuseConnection, pattern_type: PatternType) -> dict:
    return {
        "session_id": session_manager.current_session_id,
        "state": session_manager.session_state,
        "device_source": muse_connection.active_source or session_manager.session_config.get("device_source") or settings.muse_device_source,
        "board_id": int(muse_connection.board_id),
        "age": session_manager.session_config.get("age"),
        "gender": session_manager.session_config.get("gender"),
        "sampling_rate": int(muse_connection.sampling_rate or settings.muse_sampling_rate),
        "channel_count": int(muse_connection.eeg_channels or 0),
        "window_size": settings.muse_window_size,
        "update_interval": settings.eeg_update_interval,
        "pattern_type": pattern_type.value,
        "signal_sensitivity": session_manager.session_config.get("signal_sensitivity"),
        "noise_control": session_manager.session_config.get("noise_control"),
        "osc_enabled": settings.osc_enabled,
        "osc_host": settings.osc_host,
        "osc_port": settings.osc_port,
        "osc_stream_address": settings.osc_stream_address,
    }


def _log_empty_read(muse_connection: MuseConnection, empty_reads: int, last_no_data_log: float) -> float:
    now = time.monotonic()
    if now - last_no_data_log >= 5.0:
        logger.info(
            "Stream waiting for EEG samples session_id=%s empty_reads=%s diagnostics=%s",
            session_manager.current_session_id,
            empty_reads,
            muse_connection.get_runtime_diagnostics(),
        )
        return now

    return last_no_data_log


def _log_feature_failure(eeg_data, feature_failures: int, last_no_features_log: float) -> float:
    now = time.monotonic()
    if now - last_no_features_log >= 5.0:
        shape = getattr(eeg_data, "shape", None)
        logger.info(
            "Stream received EEG data but feature extraction returned no result session_id=%s feature_failures=%s shape=%s sampling_rate=%s",
            session_manager.current_session_id,
            feature_failures,
            shape,
            processor.sampling_rate,
        )
        return now

    return last_no_features_log


def _start_stream_runtime(muse_connection: MuseConnection) -> tuple[PatternType, dict]:
    if not muse_connection.is_connected():
        muse_connection.connect()

    logger.info(
        "Stream loop started session_id=%s source=%s board_id=%s",
        session_manager.current_session_id,
        muse_connection.active_source or session_manager.session_config.get("device_source") or settings.muse_device_source,
        muse_connection.board_id,
    )

    processor.sampling_rate = int(muse_connection.sampling_rate or settings.muse_sampling_rate)
    selected_pattern = _get_selected_pattern()
    stream_config = _build_stream_config(muse_connection, selected_pattern)
    session_manager.set_state(SessionState.RUNNING)
    osc_sender.send_fields({"active": 1})
    return selected_pattern, stream_config


def _build_stream_message(features, emotion_result, pattern_params, selected_pattern: PatternType, stream_config: dict) -> dict:
    stream_config["state"] = session_manager.session_state
    return {
        "timestamp": float(time.time()),
        "alpha": float(features["alpha"]),
        "beta": float(features["beta"]),
        "gamma": float(features["gamma"]),
        "theta": float(features["theta"]),
        "delta": float(features["delta"]),
        "signal_quality": 1.0,
        "emotion": emotion_result.emotion.value,
        "confidence": float(emotion_result.confidence),
        "mindfulness": (
            float(emotion_result.mindfulness)
            if emotion_result.mindfulness is not None
            else None
        ),
        "restfulness": (
            float(emotion_result.restfulness)
            if emotion_result.restfulness is not None
            else None
        ),
        "pattern_seed": int(pattern_params.pattern_seed),
        "pattern_complexity": float(pattern_params.complexity),
        "color_palette": [str(color) for color in pattern_params.color_palette],
        "config": stream_config,
        "age": stream_config.get("age"),
        "gender": stream_config.get("gender"),
        "pattern_type": selected_pattern.value,
        "active": 1,
    }


def _log_frame_progress(message: dict, frames_emitted: int, last_progress_log: float) -> float:
    now = time.monotonic()
    if frames_emitted == 1 or now - last_progress_log >= 5.0:
        logger.info(
            "Stream produced frame %s session_id=%s emotion=%s confidence=%.3f",
            frames_emitted,
            session_manager.current_session_id,
            message["emotion"],
            message["confidence"],
        )
        return now

    return last_progress_log


def _handle_stream_error(exc: Exception, frames_emitted: int):
    logger.exception(
        "Stream error for session_id=%s after %s frames: %s",
        session_manager.current_session_id,
        frames_emitted,
        exc,
    )
    session_manager.set_state(SessionState.IDLE)
    if session_manager.muse_connection is not None:
        session_manager.muse_connection.disconnect()
        session_manager.muse_connection = None
    session_manager.clear_latest_stream_message()


def _process_stream_iteration(
    muse_connection: MuseConnection,
    selected_pattern: PatternType,
    stream_config: dict,
    last_no_data_log: float,
    last_no_features_log: float,
    last_progress_log: float,
    empty_reads: int,
    feature_failures: int,
    frames_emitted: int,
) -> tuple[float, float, float, int, int, int]:
    eeg_data = muse_connection.get_eeg_data(window_size=settings.muse_window_size)
    if eeg_data is None:
        empty_reads += 1
        last_no_data_log = _log_empty_read(muse_connection, empty_reads, last_no_data_log)
        time.sleep(0.1)
        return (
            last_no_data_log,
            last_no_features_log,
            last_progress_log,
            empty_reads,
            feature_failures,
            frames_emitted,
        )

    features = processor.extract_features(eeg_data)
    if features is None:
        feature_failures += 1
        last_no_features_log = _log_feature_failure(eeg_data, feature_failures, last_no_features_log)
        time.sleep(0.1)
        return (
            last_no_data_log,
            last_no_features_log,
            last_progress_log,
            empty_reads,
            feature_failures,
            frames_emitted,
        )

    emotion_result = emotion_model.predict(features)
    session_manager.add_emotion(emotion_result.emotion.value)
    pattern_params = pattern_mapper.map_pattern(
        emotion=emotion_result.emotion,
        eeg_features=features,
        selected_pattern=selected_pattern,
    )
    message = _build_stream_message(
        features,
        emotion_result,
        pattern_params,
        selected_pattern,
        stream_config,
    )

    session_manager.set_latest_stream_message(message)
    osc_sender.send_payload(message)
    frames_emitted += 1
    last_progress_log = _log_frame_progress(message, frames_emitted, last_progress_log)
    time.sleep(settings.eeg_update_interval)
    return (
        last_no_data_log,
        last_no_features_log,
        last_progress_log,
        empty_reads,
        feature_failures,
        frames_emitted,
    )


def _run_stream_loop():
    muse_connection = _get_or_create_muse_connection()
    frames_emitted = 0
    last_progress_log = 0.0
    last_no_data_log = 0.0
    last_no_features_log = 0.0
    empty_reads = 0
    feature_failures = 0

    try:
        selected_pattern, stream_config = _start_stream_runtime(muse_connection)

        while session_manager.is_active() and not session_manager.stream_stop_event.is_set():
            (
                last_no_data_log,
                last_no_features_log,
                last_progress_log,
                empty_reads,
                feature_failures,
                frames_emitted,
            ) = _process_stream_iteration(
                muse_connection,
                selected_pattern,
                stream_config,
                last_no_data_log,
                last_no_features_log,
                last_progress_log,
                empty_reads,
                feature_failures,
                frames_emitted,
            )
    except Exception as exc:
        _handle_stream_error(exc, frames_emitted)
    finally:
        osc_sender.send_fields({"active": 0})
        session_manager.mark_stream_stopped()
        logger.info(
            "Stream loop stopped session_id=%s emitted_frames=%s empty_reads=%s feature_failures=%s state=%s",
            session_manager.current_session_id,
            frames_emitted,
            empty_reads,
            feature_failures,
            session_manager.session_state,
        )


def start_streaming() -> bool:
    started = session_manager.start_stream_thread(_run_stream_loop)
    logger.info(
        "Stream thread start requested session_id=%s started=%s",
        session_manager.current_session_id,
        started,
    )
    return started