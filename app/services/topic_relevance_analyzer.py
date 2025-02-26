import re
import math
import nltk
from collections import Counter
from typing import Dict, Any, List, Set, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nltk_data"
)
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path
nltk.data.path.append(nltk_data_dir)

# Download required NLTK resources
try:
    # Download punkt tokenizer
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    logger.info("NLTK punkt tokenizer download complete")
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")
    logger.warning("Will use simple tokenization as fallback")


class TopicRelevanceAnalyzer:
    """
    Analyzes the relevance of article content to predefined topics of interest.
    - Topic matching: Keyword-based matching against predefined interest areas
    - Top topics: Identification of 3 most relevant topics per article
    - Relevance scoring: Weighted scoring based on keyword density and importance
    """

    def __init__(self, topics_file: str = None):
        """
        Initialize the topic relevance analyzer.

        Args:
            topics_file: Path to JSON file containing topic definitions
        """
        # Default topics if no file is provided
        self.default_topics = {
            # English topics
            "technology": {
                "keywords": [
                    "software",
                    "hardware",
                    "programming",
                    "algorithm",
                    "computer",
                    "technology",
                    "digital",
                    "internet",
                    "app",
                    "application",
                    "mobile",
                    "device",
                    "code",
                    "data",
                    "cloud",
                    "ai",
                    "artificial intelligence",
                    "machine learning",
                    "neural network",
                    "blockchain",
                    "crypto",
                    "cybersecurity",
                ],
                "weight": 1.0,
            },
            # Add Polish topics
            "technologia": {
                "keywords": [
                    "oprogramowanie",
                    "sprzęt",
                    "programowanie",
                    "algorytm",
                    "komputer",
                    "technologia",
                    "cyfrowy",
                    "internet",
                    "aplikacja",
                    "mobilny",
                    "urządzenie",
                    "kod",
                    "dane",
                    "chmura",
                    "sztuczna inteligencja",
                    "uczenie maszynowe",
                    "sieć neuronowa",
                    "blockchain",
                    "krypto",
                    "cyberbezpieczeństwo",
                ],
                "weight": 1.0,
            },
            "finanse": {
                "keywords": [
                    "finanse",
                    "pieniądze",
                    "waluta",
                    "bank",
                    "kredyt",
                    "pożyczka",
                    "inwestycja",
                    "oszczędności",
                    "budżet",
                    "podatek",
                    "ekonomia",
                    "giełda",
                    "akcje",
                    "obligacje",
                    "fundusz",
                    "emerytura",
                    "ubezpieczenie",
                    "płatność",
                    "transakcja",
                    "konto",
                    "kapitał",
                    "zysk",
                    "strata",
                ],
                "weight": 1.0,
            },
            "zdrowie": {
                "keywords": [
                    "zdrowie",
                    "medycyna",
                    "lekarz",
                    "pacjent",
                    "choroba",
                    "leczenie",
                    "szpital",
                    "klinika",
                    "diagnoza",
                    "terapia",
                    "lek",
                    "farmacja",
                    "dieta",
                    "odżywianie",
                    "fitness",
                    "ćwiczenia",
                    "wellness",
                    "profilaktyka",
                    "badanie",
                    "operacja",
                    "rehabilitacja",
                    "psychologia",
                    "psychiatria",
                ],
                "weight": 1.0,
            },
            # Keep other English topics...
        }

        # Load topics from file or use defaults
        if topics_file and os.path.exists(topics_file):
            try:
                with open(topics_file, "r") as f:
                    self.topics = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading topics file: {str(e)}")
                self.topics = self.default_topics
        else:
            self.topics = self.default_topics

        # Common stop words in multiple languages (expanded)
        self.stop_words = {
            # English stop words
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "about",
            "of",
            "from",
            "as",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "he",
            "she",
            "his",
            "her",
            "we",
            "our",
            "you",
            "your",
            # Polish stop words (expanded)
            "w",
            "i",
            "na",
            "z",
            "do",
            "że",
            "to",
            "o",
            "dla",
            "jest",
            "nie",
            "się",
            "od",
            "przez",
            "po",
            "jak",
            "co",
            "lub",
            "aby",
            "przy",
            "tak",
            "który",
            "która",
            "które",
            "gdy",
            "być",
            "ten",
            "ta",
            "te",
            "tego",
            "tej",
            "tych",
            "tym",
            "temu",
            "jako",
            "tylko",
            "już",
            "też",
            "można",
            "ma",
            "był",
            "była",
            "było",
            "będzie",
            "są",
            "ich",
            "jego",
            "jej",
            "mnie",
            "mi",
            "moje",
            "twoje",
            "swoje",
            "nasz",
            "wasz",
            "a",
            "ale",
            "więc",
            "bo",
            "gdyż",
            "ponieważ",
            "oraz",
            "czy",
            "kiedy",
            "gdzie",
            "kto",
            "co",
            "który",
            "jaki",
            "czyj",
            "ile",
            "skąd",
            "dokąd",
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze the topic relevance of the given content.

        Args:
            content: The text content to analyze

        Returns:
            Dictionary containing topic relevance metrics and normalized score
        """
        if not content or len(content.strip()) < 100:
            return {
                "top_topics": [],
                "topic_matches": {},
                "normalized_score": 5.0,  # Default middle score for insufficient content
            }

        # Detect if content is likely Polish
        is_polish = self._is_likely_polish(content)

        # Tokenize content with language-aware tokenization
        try:
            # Use language-specific word pattern
            if is_polish:
                # Include Polish characters in word pattern
                words = [
                    w.lower()
                    for w in re.findall(r"\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b", content)
                ]
            else:
                words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", content)]
        except Exception as e:
            logger.warning(
                f"Tokenization failed: {str(e)}, using fallback tokenization"
            )
            # Simple fallback tokenization
            words = [
                w.lower() for w in re.findall(r"\b\w+\b", content)
            ]  # \w matches any word character

        # Filter out stop words and short words
        filtered_words = [
            word
            for word in words
            if word.isalpha() and word not in self.stop_words and len(word) > 2
        ]

        # Create a word frequency counter
        word_freq = Counter(filtered_words)

        # Calculate topic matches
        topic_matches = self._calculate_topic_matches(word_freq)

        # Get top topics
        top_topics = self._get_top_topics(topic_matches, limit=3)

        # Calculate normalized score
        normalized_score = self._calculate_normalized_score(topic_matches)

        return {
            "top_topics": top_topics,
            "topic_matches": topic_matches,
            "normalized_score": round(normalized_score, 2),
        }

    def _is_likely_polish(self, content: str) -> bool:
        """
        Detect if content is likely Polish based on character frequency.

        Args:
            content: The text content to analyze

        Returns:
            True if content is likely Polish, False otherwise
        """
        # Polish-specific characters
        polish_chars = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

        # Count Polish characters
        polish_char_count = sum(1 for char in content if char in polish_chars)

        # If more than 0.5% of characters are Polish-specific, consider it Polish
        return polish_char_count > len(content) * 0.005

    def _calculate_topic_matches(self, word_freq: Counter) -> Dict[str, float]:
        """
        Calculate how well the content matches each topic.

        Args:
            word_freq: Counter of word frequencies in the content

        Returns:
            Dictionary mapping topic names to match scores
        """
        topic_matches = {}
        total_words = sum(word_freq.values())

        if total_words == 0:
            return {}

        for topic_name, topic_data in self.topics.items():
            keywords = topic_data["keywords"]
            weight = topic_data["weight"]

            # Count keyword matches
            match_count = 0
            for keyword in keywords:
                # Handle multi-word keywords
                if " " in keyword:
                    # Simple check for multi-word phrases
                    if keyword.lower() in " ".join(word_freq.elements()).lower():
                        match_count += 5  # Give higher weight to multi-word matches
                else:
                    match_count += word_freq[keyword]

            # Calculate match score as percentage of content
            match_score = (match_count / total_words) * 100 * weight

            # Store if there's any match
            if match_score > 0:
                topic_matches[topic_name] = round(match_score, 2)

        return topic_matches

    def _get_top_topics(
        self, topic_matches: Dict[str, float], limit: int = 3
    ) -> List[str]:
        """
        Get the top N topics with highest match scores.

        Args:
            topic_matches: Dictionary mapping topic names to match scores
            limit: Maximum number of top topics to return

        Returns:
            List of top topic names
        """
        return [
            topic
            for topic, _ in sorted(
                topic_matches.items(), key=lambda x: x[1], reverse=True
            )[:limit]
        ]

    def _calculate_normalized_score(self, topic_matches: Dict[str, float]) -> float:
        """
        Calculate a normalized topic relevance score on a scale of 1-10.

        Args:
            topic_matches: Dictionary mapping topic names to match scores

        Returns:
            Normalized score from 1-10
        """
        if not topic_matches:
            return 5.0  # Default middle score

        # Get the highest match score
        max_score = max(topic_matches.values()) if topic_matches else 0

        # Calculate the number of topics with significant matches
        significant_topics = sum(1 for score in topic_matches.values() if score > 1.0)

        # Combine max score and topic diversity for final score
        # - Max score contributes 70% (how strongly it matches the best topic)
        # - Topic diversity contributes 30% (how many topics it matches)

        # Normalize max score to 0-10 scale (assuming max possible is around 20%)
        max_score_normalized = min(10, max_score / 2)

        # Normalize topic diversity (assuming max of 5 significant topics is ideal)
        diversity_normalized = min(10, significant_topics * 2)

        # Weighted combination
        final_score = (max_score_normalized * 0.7) + (diversity_normalized * 0.3)

        # Ensure score is in 1-10 range
        return max(1, min(10, final_score))
