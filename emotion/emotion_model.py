import logging
import numpy as np
from typing import Dict
from brainflow.ml_model import (
    BrainFlowClassifiers,
    BrainFlowMetrics,
    BrainFlowModelParams,
    MLModel,
)

from models.schemas import EmotionResult, EmotionType
from emotion.emotion_mapping import EmotionMapper


logger = logging.getLogger("sentio.emotion")


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
        self.metric_models = self._load_brainflow_models()

        if model_path:
            self._load_model(model_path)

    def _load_brainflow_models(self) -> Dict[str, MLModel]:
        models = {}

        for name, metric in {
            "mindfulness": BrainFlowMetrics.MINDFULNESS,
            "restfulness": BrainFlowMetrics.RESTFULNESS,
        }.items():
            try:
                model_params = BrainFlowModelParams(
                    metric.value,
                    BrainFlowClassifiers.DEFAULT_CLASSIFIER.value,
                )
                model = MLModel(model_params)
                model.prepare()
                models[name] = model
            except Exception as exc:
                logger.warning("Failed to prepare BrainFlow %s model: %s", name, exc)

        return models

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

        metrics = self._predict_brainflow_metrics(eeg_features)
        result = self.mapper.detect_emotion(
            eeg_features,
            mindfulness=metrics.get("mindfulness"),
            restfulness=metrics.get("restfulness"),
        )

        return EmotionResult(
            emotion=result["emotion"],
            confidence=result["confidence"],
            mindfulness=metrics.get("mindfulness"),
            restfulness=metrics.get("restfulness"),
        )

    def _predict_brainflow_metrics(self, eeg_features: Dict[str, float]) -> Dict[str, float]:
        feature_vector = eeg_features.get("brainflow_feature_vector")
        if not feature_vector or not self.metric_models:
            return {}

        feature_array = np.asarray(feature_vector, dtype=np.float64)
        if feature_array.size != 10:
            return {}

        metrics = {}

        for name, model in self.metric_models.items():
            try:
                prediction = np.asarray(model.predict(feature_array)).reshape(-1)
                if prediction.size == 0:
                    continue
                metrics[name] = round(float(np.clip(prediction[0], 0.0, 1.0)), 3)
            except Exception as exc:
                logger.warning("Failed to calculate BrainFlow %s score: %s", name, exc)

        return metrics

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

    def __del__(self):
        for model in getattr(self, "metric_models", {}).values():
            try:
                model.release()
            except Exception:
                pass