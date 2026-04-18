from typing import Dict, Optional
from models.schemas import EmotionType


class EmotionMapper:
    """
    Maps EEG band features to emotional states.
    Uses a rule-based heuristic suitable for real-time demos.
    """

    def __init__(self):
        self.focus_threshold = 0.65
        self.rest_threshold = 0.65
        self.stress_threshold = 0.35

    def detect_emotion(
        self,
        eeg_features: Dict[str, float],
        mindfulness: Optional[float] = None,
        restfulness: Optional[float] = None,
    ) -> Dict:
        """
        Determine emotional state from BrainFlow metrics and EEG band ratios.
        """

        alpha = eeg_features.get("alpha", 0.0)
        beta = eeg_features.get("beta", 0.0)
        gamma = eeg_features.get("gamma", 0.0)

        emotion = EmotionType.calm
        confidence = 0.5

        if (
            mindfulness is not None
            and restfulness is not None
            and mindfulness >= self.focus_threshold
            and restfulness >= self.rest_threshold
        ):
            emotion = EmotionType.calm
            confidence = (mindfulness + restfulness) / 2
        elif restfulness is not None and restfulness >= self.rest_threshold and alpha >= beta:
            emotion = EmotionType.relaxed
            confidence = max(restfulness, alpha)
        elif mindfulness is not None and mindfulness >= self.focus_threshold:
            emotion = EmotionType.focused
            confidence = max(mindfulness, beta)
        elif gamma > beta and gamma > alpha and (restfulness is None or restfulness < self.rest_threshold):
            emotion = EmotionType.excited
            confidence = gamma
        elif restfulness is not None and restfulness < self.stress_threshold and beta >= alpha:
            emotion = EmotionType.stressed
            confidence = max(1.0 - restfulness, beta)
        elif alpha > beta:
            emotion = EmotionType.calm
            confidence = alpha
        elif beta > alpha:
            emotion = EmotionType.focused
            confidence = beta

        return {
            "emotion": emotion,
            "confidence": round(min(max(confidence, 0.0), 1.0), 3)
        }