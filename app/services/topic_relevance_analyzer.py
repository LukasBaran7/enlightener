import re
import math
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import Dict, Any, List, Set, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nltk_data"
    )
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)


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
                    "tech",
                    "innovation",
                    "startup",
                    "automation",
                    "robot",
                    "iot",
                    "web3",
                ],
                "weight": 1.0,
            },
            "business": {
                "keywords": [
                    "business",
                    "company",
                    "corporation",
                    "startup",
                    "entrepreneur",
                    "market",
                    "finance",
                    "investment",
                    "stock",
                    "economy",
                    "economic",
                    "industry",
                    "commercial",
                    "trade",
                    "venture",
                    "capital",
                    "profit",
                    "revenue",
                    "growth",
                    "strategy",
                    "management",
                    "leadership",
                    "ceo",
                    "executive",
                    "board",
                    "shareholder",
                    "stakeholder",
                    "merger",
                    "acquisition",
                ],
                "weight": 1.0,
            },
            "science": {
                "keywords": [
                    "science",
                    "scientific",
                    "research",
                    "experiment",
                    "laboratory",
                    "discovery",
                    "theory",
                    "hypothesis",
                    "evidence",
                    "data",
                    "analysis",
                    "physics",
                    "chemistry",
                    "biology",
                    "astronomy",
                    "geology",
                    "medicine",
                    "neuroscience",
                    "genetics",
                    "molecular",
                    "quantum",
                    "particle",
                    "cell",
                    "organism",
                    "ecosystem",
                    "climate",
                    "environment",
                    "sustainability",
                ],
                "weight": 1.0,
            },
            "health": {
                "keywords": [
                    "health",
                    "healthcare",
                    "medical",
                    "medicine",
                    "doctor",
                    "hospital",
                    "patient",
                    "treatment",
                    "therapy",
                    "disease",
                    "condition",
                    "symptom",
                    "diagnosis",
                    "prevention",
                    "wellness",
                    "fitness",
                    "nutrition",
                    "diet",
                    "exercise",
                    "mental health",
                    "psychology",
                    "psychiatry",
                    "pharmaceutical",
                    "drug",
                    "vaccine",
                    "immunity",
                    "surgery",
                    "recovery",
                    "chronic",
                    "acute",
                ],
                "weight": 1.0,
            },
            "politics": {
                "keywords": [
                    "politics",
                    "political",
                    "government",
                    "policy",
                    "election",
                    "vote",
                    "democracy",
                    "democratic",
                    "republican",
                    "liberal",
                    "conservative",
                    "progressive",
                    "legislation",
                    "law",
                    "regulation",
                    "congress",
                    "senate",
                    "parliament",
                    "president",
                    "prime minister",
                    "diplomat",
                    "foreign policy",
                    "domestic policy",
                    "campaign",
                    "candidate",
                    "party",
                    "constitution",
                ],
                "weight": 1.0,
            },
            "finance": {
                "keywords": [
                    "finance",
                    "financial",
                    "money",
                    "banking",
                    "bank",
                    "investment",
                    "investor",
                    "fund",
                    "stock",
                    "bond",
                    "market",
                    "trading",
                    "trader",
                    "portfolio",
                    "asset",
                    "liability",
                    "equity",
                    "debt",
                    "credit",
                    "loan",
                    "mortgage",
                    "interest",
                    "dividend",
                    "capital",
                    "cash",
                    "currency",
                    "exchange",
                    "inflation",
                    "deflation",
                    "recession",
                    "economy",
                    "tax",
                ],
                "weight": 1.0,
            },
            "education": {
                "keywords": [
                    "education",
                    "educational",
                    "school",
                    "university",
                    "college",
                    "student",
                    "teacher",
                    "professor",
                    "academic",
                    "learning",
                    "teaching",
                    "curriculum",
                    "course",
                    "class",
                    "lecture",
                    "study",
                    "research",
                    "knowledge",
                    "skill",
                    "literacy",
                    "scholarship",
                    "degree",
                    "diploma",
                    "certificate",
                    "training",
                    "development",
                    "pedagogy",
                    "instruction",
                    "classroom",
                    "online learning",
                ],
                "weight": 1.0,
            },
            "entertainment": {
                "keywords": [
                    "entertainment",
                    "movie",
                    "film",
                    "television",
                    "tv",
                    "show",
                    "series",
                    "actor",
                    "actress",
                    "director",
                    "producer",
                    "celebrity",
                    "star",
                    "fame",
                    "music",
                    "musician",
                    "artist",
                    "band",
                    "concert",
                    "performance",
                    "theater",
                    "theatre",
                    "stage",
                    "comedy",
                    "drama",
                    "streaming",
                    "netflix",
                    "disney",
                    "hollywood",
                    "bollywood",
                    "game",
                    "gaming",
                    "video game",
                    "esports",
                ],
                "weight": 1.0,
            },
        }

        # Load custom topics if file is provided
        self.topics = self._load_topics(topics_file)

        # Common stop words to filter out
        self.stop_words = set(
            [
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
            ]
        )

    def _load_topics(self, topics_file: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Load topics from file or use defaults.

        Args:
            topics_file: Path to JSON file containing topic definitions

        Returns:
            Dictionary of topics with keywords and weights
        """
        if topics_file and os.path.exists(topics_file):
            try:
                with open(topics_file, "r") as f:
                    topics = json.load(f)
                logger.info(f"Loaded {len(topics)} topics from {topics_file}")
                return topics
            except Exception as e:
                logger.warning(f"Failed to load topics from {topics_file}: {str(e)}")
                logger.warning("Using default topics instead")

        logger.info(f"Using {len(self.default_topics)} default topics")
        return self.default_topics

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

        # Tokenize and preprocess content
        try:
            words = word_tokenize(content.lower())
        except Exception as e:
            logger.warning(
                f"NLTK tokenization failed: {str(e)}, using fallback tokenization"
            )
            # Simple fallback tokenization
            words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]+\b", content)]

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
