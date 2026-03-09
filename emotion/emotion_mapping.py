from typing import Dict
from models.schemas import EmotionType


class EmotionMapper:
    """
    Maps EEG band features to emotional states.
    Uses a rule-based heuristic suitable for real-time demos.
    """

    def __init__(self):
        pass

    def detect_emotion(self, eeg_features: Dict[str, float]) -> Dict:
        """
        Determine emotional state based on EEG band ratios.
        """

        alpha = eeg_features.get("alpha", 0.0)
        beta = eeg_features.get("beta", 0.0)
        gamma = eeg_features.get("gamma", 0.0)
        theta = eeg_features.get("theta", 0.0)
        delta = eeg_features.get("delta", 0.0)

        emotion = EmotionType.calm
        confidence = 0.5

        # Relaxed / Calm
        if alpha > beta and alpha > gamma:
            emotion = EmotionType.relaxed
            confidence = min(alpha, 1.0)

        # Focused / Concentration
        elif beta > alpha and beta > gamma:
            emotion = EmotionType.focused
            confidence = min(beta, 1.0)

        # High stimulation / excitement
        elif gamma > beta and gamma > alpha:
            emotion = EmotionType.excited
            confidence = min(gamma, 1.0)

        # Calm baseline
        if alpha > 0.4 and beta < 0.3:
            emotion = EmotionType.calm
            confidence = alpha

        # Stress indicator
        if beta > 0.6 and gamma > 0.4:
            emotion = EmotionType.stressed
            confidence = beta

        return {
            "emotion": emotion,
            "confidence": round(confidence, 3)
        }