import numpy as np
from typing import Dict

from models.schemas import EmotionResult, EmotionType
from emotion.emotion_mapping import EmotionMapper


class EmotionModel:
    """
    Emotion detection model for Sentio.

    Supports:
    - rule-based inference
    - optional ML model
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self.mapper = EmotionMapper()

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        Load trained ML model if available.
        """
        try:
            import joblib
            self.model = joblib.load(model_path)
            print("Emotion ML model loaded.")
        except Exception as e:
            print("Failed to load ML model, using rule-based mapping.", e)
            self.model = None

    def predict(self, eeg_features: Dict[str, float]) -> EmotionResult:
        """
        Predict emotional state from EEG features.
        """

        if self.model:
            return self._predict_ml(eeg_features)

        return self._predict_rule_based(eeg_features)

    def _predict_rule_based(self, eeg_features: Dict[str, float]) -> EmotionResult:
        """
        Use heuristic emotion mapping.
        """

        result = self.mapper.detect_emotion(eeg_features)

        return EmotionResult(
            emotion=result["emotion"],
            confidence=result["confidence"]
        )

    def _predict_ml(self, eeg_features: Dict[str, float]) -> EmotionResult:
        """
        Predict emotion using trained ML model.
        """

        feature_vector = np.array([
            eeg_features.get("alpha", 0),
            eeg_features.get("beta", 0),
            eeg_features.get("gamma", 0),
            eeg_features.get("theta", 0),
            eeg_features.get("delta", 0)
        ]).reshape(1, -1)

        prediction = self.model.predict(feature_vector)[0]

        if hasattr(self.model, "predict_proba"):
            confidence = float(np.max(self.model.predict_proba(feature_vector)))
        else:
            confidence = 0.7

        return EmotionResult(
            emotion=EmotionType(prediction),
            confidence=round(confidence, 3)
        )