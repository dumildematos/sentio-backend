import time

from config import settings
from eeg.muse_connection import MuseConnection
from eeg.signal_processing import SignalProcessor
from emotion.emotion_model import EmotionModel
from models.schemas import PatternType
from patterns.pattern_mapper import PatternMapper
from services.osc_sender import OscStreamSender
from services.session_manager import SessionState, session_manager

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
            mac_address=session_manager.session_config.get("mac_address") or settings.muse_mac_address
        )
        session_manager.muse_connection = muse_connection
    return muse_connection


def _build_stream_config(muse_connection: MuseConnection, pattern_type: PatternType) -> dict:
    return {
        "session_id": session_manager.current_session_id,
        "state": session_manager.session_state,
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


def _run_stream_loop():
    muse_connection = _get_or_create_muse_connection()

    try:
        if not muse_connection.inlet and not muse_connection.connect():
            raise RuntimeError("Unable to connect to BlueMuse EEG stream.")

        processor.sampling_rate = int(muse_connection.sampling_rate or settings.muse_sampling_rate)
        selected_pattern = _get_selected_pattern()
        stream_config = _build_stream_config(muse_connection, selected_pattern)
        session_manager.set_state(SessionState.RUNNING)
        osc_sender.send_fields({"active": 1})

        while session_manager.is_active() and not session_manager.stream_stop_event.is_set():
            eeg_data = muse_connection.get_eeg_data(window_size=settings.muse_window_size)
            if eeg_data is None:
                time.sleep(0.1)
                continue

            features = processor.extract_features(eeg_data)
            if features is None:
                time.sleep(0.1)
                continue

            emotion_result = emotion_model.predict(features)
            session_manager.add_emotion(emotion_result.emotion.value)
            pattern_params = pattern_mapper.map_pattern(
                emotion=emotion_result.emotion,
                eeg_features=features,
                selected_pattern=selected_pattern,
            )
            stream_config["state"] = session_manager.session_state

            message = {
                "timestamp": float(time.time()),
                "alpha": float(features["alpha"]),
                "beta": float(features["beta"]),
                "gamma": float(features["gamma"]),
                "theta": float(features["theta"]),
                "delta": float(features["delta"]),
                "signal_quality": 1.0,
                "emotion": emotion_result.emotion.value,
                "confidence": float(emotion_result.confidence),
                "pattern_seed": int(pattern_params.pattern_seed),
                #"pattern_type": pattern_params.pattern_type.value,
                "pattern_complexity": float(pattern_params.complexity),
                "color_palette": [str(color) for color in pattern_params.color_palette],
                "config": stream_config,
                "age": stream_config.get("age"),
                "pattern_type": selected_pattern.value,
                "active": 1,
            }

            session_manager.set_latest_stream_message(message)
            osc_sender.send_payload(message)
            time.sleep(settings.eeg_update_interval)
    except Exception as exc:
        print(f"Stream error: {exc}")
        session_manager.set_state(SessionState.IDLE)
        if session_manager.muse_connection is not None:
            session_manager.muse_connection.disconnect()
            session_manager.muse_connection = None
        session_manager.clear_latest_stream_message()
    finally:
        osc_sender.send_fields({"active": 0})
        session_manager.mark_stream_stopped()


def start_streaming() -> bool:
    return session_manager.start_stream_thread(_run_stream_loop)