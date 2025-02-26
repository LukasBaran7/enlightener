import re
import math
from typing import Dict, Any, List, Tuple
import textstat


class ReadabilityAnalyzer:
    """
    Analyzes the readability of article content using various metrics:
    - Flesch Reading Ease
    - SMOG Index
    - Coleman-Liau Index
    - Automated Readability Index
    - Complexity classification
    """

    def __init__(self):
        """Initialize the readability analyzer."""
        pass

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze the readability of the given content.

        Args:
            content: The text content to analyze

        Returns:
            Dictionary containing readability metrics and normalized score
        """
        if not content or len(content.strip()) < 100:
            return {
                "flesch_reading_ease": 0,
                "smog_index": 0,
                "coleman_liau_index": 0,
                "automated_readability_index": 0,
                "complexity_level": "Unknown",
                "normalized_score": 5.0,  # Default middle score for insufficient content
            }

        # Calculate readability metrics
        flesch_reading_ease = textstat.flesch_reading_ease(content)
        smog_index = textstat.smog_index(content)
        coleman_liau_index = textstat.coleman_liau_index(content)
        automated_readability_index = textstat.automated_readability_index(content)

        # Determine complexity level
        complexity_level = self._determine_complexity_level(
            flesch_reading_ease,
            smog_index,
            coleman_liau_index,
            automated_readability_index,
        )

        # Calculate normalized score (1-10)
        normalized_score = self._calculate_normalized_score(
            flesch_reading_ease,
            smog_index,
            coleman_liau_index,
            automated_readability_index,
        )

        return {
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "smog_index": round(smog_index, 2),
            "coleman_liau_index": round(coleman_liau_index, 2),
            "automated_readability_index": round(automated_readability_index, 2),
            "complexity_level": complexity_level,
            "normalized_score": round(normalized_score, 2),
        }

    def _determine_complexity_level(
        self,
        flesch_reading_ease: float,
        smog_index: float,
        coleman_liau_index: float,
        automated_readability_index: float,
    ) -> str:
        """
        Determine the complexity level based on readability metrics.

        Args:
            flesch_reading_ease: Flesch Reading Ease score
            smog_index: SMOG Index
            coleman_liau_index: Coleman-Liau Index
            automated_readability_index: Automated Readability Index

        Returns:
            Complexity level as string: Basic, Intermediate, Advanced, or Expert
        """
        # Convert Flesch Reading Ease to grade level (higher score = easier to read)
        # So we invert the scale for consistency with other metrics
        flesch_grade = 0
        if flesch_reading_ease >= 90:
            flesch_grade = 5  # 5th grade
        elif flesch_reading_ease >= 80:
            flesch_grade = 6  # 6th grade
        elif flesch_reading_ease >= 70:
            flesch_grade = 7  # 7th grade
        elif flesch_reading_ease >= 60:
            flesch_grade = 8.5  # 8-9th grade
        elif flesch_reading_ease >= 50:
            flesch_grade = 10.5  # 10-11th grade
        elif flesch_reading_ease >= 30:
            flesch_grade = 13  # College
        else:
            flesch_grade = 15  # College graduate

        # Average the grade levels from different metrics
        average_grade = (
            flesch_grade + smog_index + coleman_liau_index + automated_readability_index
        ) / 4

        # Map average grade to complexity level
        if average_grade < 8:
            return "Basic"
        elif average_grade < 12:
            return "Intermediate"
        elif average_grade < 16:
            return "Advanced"
        else:
            return "Expert"

    def _calculate_normalized_score(
        self,
        flesch_reading_ease: float,
        smog_index: float,
        coleman_liau_index: float,
        automated_readability_index: float,
    ) -> float:
        """
        Calculate a normalized readability score on a scale of 1-10.

        For readability, a middle score (5) is optimal - neither too simple nor too complex.

        Args:
            flesch_reading_ease: Flesch Reading Ease score
            smog_index: SMOG Index
            coleman_liau_index: Coleman-Liau Index
            automated_readability_index: Automated Readability Index

        Returns:
            Normalized score from 1-10
        """
        # Convert metrics to a common scale where higher = more complex
        # Flesch is reversed (higher = easier), so we invert it
        normalized_flesch = 100 - flesch_reading_ease

        # For the other indices, higher = more complex
        # Normalize each to 0-100 scale
        # Assuming grade level indices range from ~0 to ~20
        normalized_smog = min(100, smog_index * 5)
        normalized_coleman = min(100, coleman_liau_index * 5)
        normalized_ari = min(100, automated_readability_index * 5)

        # Average the normalized scores
        avg_complexity = (
            normalized_flesch + normalized_smog + normalized_coleman + normalized_ari
        ) / 4

        # Map to 1-10 scale with 5 as optimal (moderate complexity)
        # 0-25 complexity -> 1-4 score (too simple)
        # 25-75 complexity -> 4-7 score (optimal range)
        # 75-100 complexity -> 7-10 score (too complex)

        if avg_complexity < 25:
            # Too simple (1-4)
            return 1 + (avg_complexity / 25) * 3
        elif avg_complexity < 75:
            # Optimal range (4-7)
            return 4 + ((avg_complexity - 25) / 50) * 3
        else:
            # Too complex (7-10)
            return 7 + ((avg_complexity - 75) / 25) * 3
