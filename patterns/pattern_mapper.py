import random
from typing import Dict
from models.schemas import EmotionType, PatternType, PatternParameters


class PatternMapper:
    """
    Maps emotional states and EEG intensity into
    parameters used by the frontend generative pattern engine.
    """

    def __init__(self):
        self.emotion_color_palettes = {
            EmotionType.calm: [
                "#4F6D7A",
                "#A6C8D8",
                "#E6F1F5",
                "#2E4057"
            ],
            EmotionType.focused: [
                "#3A86FF",
                "#4361EE",
                "#4CC9F0",
                "#1D3557"
            ],
            EmotionType.relaxed: [
                "#A8DADC",
                "#F1FAEE",
                "#B7E4C7",
                "#52B788"
            ],
            EmotionType.excited: [
                "#FF006E",
                "#FB5607",
                "#FFBE0B",
                "#FF7F51"
            ],
            EmotionType.stressed: [
                "#6A040F",
                "#9D0208",
                "#D00000",
                "#370617"
            ],
        }

    def map_pattern(
        self,
        emotion: EmotionType,
        eeg_features: Dict[str, float],
        selected_pattern: PatternType
    ) -> PatternParameters:
        """
        Convert emotion + EEG data into pattern parameters.
        """

        alpha = eeg_features.get("alpha", 0.5)
        beta = eeg_features.get("beta", 0.5)
        gamma = eeg_features.get("gamma", 0.5)

        complexity = self._compute_complexity(alpha, beta, gamma)

        palette = self.emotion_color_palettes.get(
            emotion,
            ["#888888", "#CCCCCC"]
        )

        pattern_seed = random.randint(0, 100000)

        return PatternParameters(
            pattern_type=selected_pattern,
            color_palette=palette,
            complexity=complexity,
            pattern_seed=pattern_seed
        )

    def _compute_complexity(self, alpha: float, beta: float, gamma: float) -> float:
        """
        Compute pattern complexity based on EEG energy.
        Higher beta/gamma -> more energetic patterns.
        """

        energy = (alpha * 0.3) + (beta * 0.4) + (gamma * 0.3)

        complexity = min(max(energy, 0.1), 1.0)

        return complexity