import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import math

logger = logging.getLogger(__name__)


class FreshnessAnalyzer:
    """
    Analyzes the freshness of article content based on:
    - Publication date: How recent the article is
    - Temporal references: Mentions of dates, time periods, and current events
    - Content decay: How quickly the content type loses relevance over time
    """

    def __init__(self):
        """Initialize the freshness analyzer."""
        # Patterns for identifying temporal references
        self.temporal_patterns = [
            r"\b(?:today|yesterday|tomorrow)\b",
            r"\b(?:this|last|next)\s+(?:week|month|year|quarter)\b",
            r"\b(?:recent|upcoming|current|latest|new|future)\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),\s+\d{4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",  # ISO date format
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",  # MM/DD/YY or MM/DD/YYYY
        ]
        self.temporal_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.temporal_patterns
        ]

        # Content decay rates by category (half-life in days)
        # Lower values mean content becomes outdated more quickly
        self.decay_rates = {
            "news": 3,  # News becomes outdated very quickly
            "technology": 90,  # Tech content has medium-term relevance
            "science": 365,  # Scientific content stays relevant longer
            "evergreen": 1095,  # Evergreen content (like guides) stays relevant for years
            "reference": 1825,  # Reference material has very long relevance
            "default": 180,  # Default decay rate for uncategorized content
        }

    def analyze(
        self,
        content: str,
        published_date: Optional[datetime] = None,
        category: str = "default",
        current_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the freshness of the given content.

        Args:
            content: The text content to analyze
            published_date: Publication date of the article
            category: Content category for decay rate calculation
            current_date: Current date (defaults to now)

        Returns:
            Dictionary containing freshness metrics and normalized score
        """
        if current_date is None:
            current_date = datetime.now()

        if not content or len(content.strip()) < 100:
            return {
                "age_days": 0,
                "temporal_references_count": 0,
                "decay_rate": self.decay_rates.get(
                    category, self.decay_rates["default"]
                ),
                "is_recent": False,
                "normalized_score": 5.0,  # Default middle score for insufficient content
            }

        # Count temporal references
        temporal_references_count = self._count_temporal_references(content)

        # Calculate age in days
        age_days = 0
        is_recent = False
        if published_date:
            age_days = (current_date - published_date).days
            is_recent = age_days <= 7  # Consider content recent if less than a week old

        # Get decay rate for the category
        decay_rate = self.decay_rates.get(category, self.decay_rates["default"])

        # Calculate normalized score
        normalized_score = self._calculate_normalized_score(
            age_days, temporal_references_count, decay_rate
        )

        return {
            "age_days": age_days,
            "temporal_references_count": temporal_references_count,
            "decay_rate": decay_rate,
            "is_recent": is_recent,
            "normalized_score": round(normalized_score, 2),
        }

    def _count_temporal_references(self, content: str) -> int:
        """
        Count temporal references in the content.

        Args:
            content: The text content to analyze

        Returns:
            Number of temporal references found
        """
        count = 0
        for pattern in self.temporal_patterns:
            matches = pattern.findall(content)
            count += len(matches)

        return count

    def _calculate_normalized_score(
        self, age_days: int, temporal_references_count: int, decay_rate: int
    ) -> float:
        """
        Calculate a normalized freshness score on a scale of 1-10.

        Args:
            age_days: Age of the content in days
            temporal_references_count: Number of temporal references
            decay_rate: Content decay rate (half-life in days)

        Returns:
            Normalized score from 1-10
        """
        # If we don't have a publication date, base score primarily on temporal references
        if age_days == 0:
            # Scale based on temporal references (0-10 references maps to 5-8 score)
            temporal_score = 5 + min(3, temporal_references_count * 0.3)
            return temporal_score

        # Calculate age factor using exponential decay formula
        # Score decreases as age increases, with rate determined by decay_rate
        age_factor = math.exp(-math.log(2) * age_days / decay_rate)

        # Calculate temporal references factor
        # More temporal references can indicate time-sensitive content
        temporal_factor = min(1.0, temporal_references_count * 0.1)

        # Combine factors with weights
        # Age is the primary factor (80%), temporal references secondary (20%)
        combined_score = (age_factor * 0.8) + (temporal_factor * 0.2)

        # Map to 1-10 scale
        # 1 = very outdated, 10 = extremely fresh
        normalized_score = 1 + combined_score * 9

        return normalized_score
