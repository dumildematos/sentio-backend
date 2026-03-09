from typing import Any, Dict

from pythonosc.udp_client import SimpleUDPClient


class OscStreamSender:
    def __init__(self, enabled: bool, host: str, port: int, stream_address: str):
        self.enabled = enabled
        self.host = host
        self.port = port
        normalized_address = (stream_address or "/emotion").strip()
        self.stream_address = normalized_address if normalized_address.startswith("/") else f"/{normalized_address}"
        self.client = SimpleUDPClient(host, port) if enabled else None

    def send_payload(self, payload: Dict[str, Any]):
        if not self.client or not payload:
            return

        try:
            primary_payload = self._build_primary_payload(payload)
            self._send_nested(self.stream_address, primary_payload)
            print(f"address {self.stream_address}", f"payload {primary_payload}")
        except Exception as exc:
            print(f"OSC send error: {exc}")

    def send_fields(self, values: Dict[str, Any]):
        if not self.client or not values:
            return

        try:
            self._send_nested(self.stream_address, values)
            print(f"address {self.stream_address}", f"payload {values}")
        except Exception as exc:
            print(f"OSC send error: {exc}")

    def _build_primary_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = payload.get("config") or {}
        emotion = str(payload.get("emotion", ""))
        alpha = float(payload.get("alpha", 0.0) or 0.0)
        beta = float(payload.get("beta", 0.0) or 0.0)
        gamma = float(payload.get("gamma", 0.0) or 0.0)
        theta = float(payload.get("theta", 0.0) or 0.0)
        delta = float(payload.get("delta", 0.0) or 0.0)
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        signal_quality = float(payload.get("signal_quality", 0.0) or 0.0)
        pattern_seed = int(payload.get("pattern_seed", 0) or 0)
        pattern_complexity = float(payload.get("pattern_complexity", 0.0) or 0.0)
        age = payload.get("age", config.get("age"))
        gender = payload.get("gender", config.get("gender"))    
        return {
            "timestamp": float(payload.get("timestamp", 0.0) or 0.0),
            "emotion": emotion,
            "confidence": confidence,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "theta": theta,
            "delta": delta,
            "signal_quality": signal_quality,
            "pattern_seed": pattern_seed,
            "pattern_complexity": pattern_complexity,
            "age": int(age) if age is not None else None,
            "gender": str(gender) if gender is not None else None,
            "pattern_type": str(payload.get("pattern_type", config.get("pattern_type", ""))),
            "active": int(bool(payload.get("active", True))),
        }

    def _send_metadata(self, payload: Dict[str, Any]):
        config = payload.get("config") or {}
        pattern_type = self._normalize_string(payload.get("pattern_type") or config.get("pattern_type"))
        gender = self._normalize_string(payload.get("gender") or config.get("gender"))

        if pattern_type:
            self._send(f"{self.stream_address}/pattern_type", pattern_type)
        if gender:
            self._send(f"{self.stream_address}/gender", gender)

    def _send_nested(self, address: str, value: Any):
        if value is None:
            return

        if isinstance(value, dict):
            for key, nested_value in value.items():
                self._send_nested(f"{address}/{self._normalize_segment(key)}", nested_value)
            return

        if isinstance(value, list):
            if not value:
                return
            if all(self._is_osc_atom(item) for item in value):
                self._send(address, [self._normalize_atom(item) for item in value])
                return
            for index, nested_value in enumerate(value):
                self._send_nested(f"{address}/{index}", nested_value)
            return

        if self._is_osc_atom(value):
            self._send(address, self._normalize_atom(value))

    def _send(self, address: str, value: Any):
        if self.client:
            self.client.send_message(address, value)

    @staticmethod
    def _is_osc_atom(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool))

    @staticmethod
    def _normalize_atom(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, bool):
            return int(value)
        return value

    @classmethod
    def _normalize_string(cls, value: Any) -> str:
        normalized = cls._normalize_atom(value)
        return str(normalized) if normalized is not None else ""

    @staticmethod
    def _normalize_segment(segment: Any) -> str:
        return str(segment).strip().replace(" ", "_")