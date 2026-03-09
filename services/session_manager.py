import copy
import threading
import uuid
import time
from typing import Optional, Dict, Any


class SessionState:
    IDLE = "idle"
    CONNECTING = "connecting"
    CALIBRATING = "calibrating"
    RUNNING = "running"
    STOPPED = "stopped"


class SessionManager:
    """
    Manages Sentio EEG sessions.
    Handles configuration, state transitions, and session metadata.
    """

    def __init__(self):
        self.current_session_id: Optional[str] = None
        self.session_config: Dict[str, Any] = {}
        self.session_state: str = SessionState.IDLE
        self.start_time: Optional[float] = None
        self.emotion_history = []
        self.muse_connection: Optional[Any] = None
        self.latest_stream_message: Optional[Dict[str, Any]] = None
        self.stream_thread: Optional[threading.Thread] = None
        self.stream_stop_event = threading.Event()
        self._stream_lock = threading.Lock()

    def start_session(self, config: Dict[str, Any]) -> str:
        """
        Create and start a new session.
        """
        self.current_session_id = str(uuid.uuid4())
        self.session_config = copy.deepcopy(config)
        self.session_state = SessionState.CONNECTING
        self.start_time = time.time()
        self.emotion_history = []
        self.clear_latest_stream_message()

        return self.current_session_id

    def set_state(self, state: str):
        """
        Update session state.
        """
        self.session_state = state

    def stop_session(self):
        """
        Stop the active session.
        """
        self.session_state = SessionState.STOPPED
        self.request_stream_stop()
        self.wait_for_stream_stop(timeout=2.0)
        if self.muse_connection is not None:
            self.muse_connection.disconnect()
        self.current_session_id = None
        self.session_config = {}
        self.start_time = None
        self.emotion_history = []
        self.muse_connection = None
        self.clear_latest_stream_message()

    def add_emotion(self, emotion: str):
        """
        Store emotion history during session.
        """
        self.emotion_history.append({
            "emotion": emotion,
            "timestamp": time.time()
        })

    def get_session_info(self) -> Dict[str, Any]:
        """
        Return current session status.
        """
        return {
            "session_id": self.current_session_id,
            "state": self.session_state,
            "config": self.session_config,
            "start_time": self.start_time,
            "emotion_history_length": len(self.emotion_history)
        }

    def set_latest_stream_message(self, message: Dict[str, Any]):
        with self._stream_lock:
            self.latest_stream_message = copy.deepcopy(message)

    def get_latest_stream_message(self) -> Optional[Dict[str, Any]]:
        with self._stream_lock:
            if self.latest_stream_message is None:
                return None
            return copy.deepcopy(self.latest_stream_message)

    def clear_latest_stream_message(self):
        with self._stream_lock:
            self.latest_stream_message = None

    def start_stream_thread(self, target) -> bool:
        with self._stream_lock:
            if self.stream_thread is not None and self.stream_thread.is_alive():
                return False

            self.stream_stop_event.clear()
            self.stream_thread = threading.Thread(
                target=target,
                daemon=True,
                name="eeg-stream",
            )
            self.stream_thread.start()
            return True

    def request_stream_stop(self):
        self.stream_stop_event.set()

    def wait_for_stream_stop(self, timeout: float = 2.0):
        stream_thread = self.stream_thread
        if (
            stream_thread is not None
            and stream_thread.is_alive()
            and threading.current_thread() is not stream_thread
        ):
            stream_thread.join(timeout=timeout)

        if stream_thread is None or not stream_thread.is_alive():
            self.stream_thread = None

    def mark_stream_stopped(self):
        with self._stream_lock:
            self.stream_thread = None

    def is_streaming(self) -> bool:
        return self.stream_thread is not None and self.stream_thread.is_alive()

    def is_active(self) -> bool:
        """
        Check if session is currently active.
        """
        return self.session_state in [
            SessionState.CONNECTING,
            SessionState.CALIBRATING,
            SessionState.RUNNING
        ]

# Singleton instance for import
session_manager = SessionManager()