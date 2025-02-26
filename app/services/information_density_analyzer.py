import re
import math
import nltk
from collections import Counter
from typing import Dict, Any, List, Set, Tuple
import logging
import os
import string

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
    # Download punkt tokenizer directly without checking first
    nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
    logger.info("NLTK punkt tokenizer download complete")
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")
    logger.warning("Will use simple tokenization as fallback")


class InformationDensityAnalyzer:
    """
    Analyzes the information density of article content using various metrics:
    - Lexical diversity: Unique words ratio
    - Fact density: Proportion of sentences containing factual elements
    - Concept density: Identification of key concepts and their occurrence patterns
    """

    def __init__(self):
        """Initialize the information density analyzer."""
        # Patterns for identifying factual content (including Polish patterns)
        self.fact_patterns = [
            # English patterns
            r"\b\d+(?:\.\d+)?%\b",  # Percentages
            r"\b\d+(?:,\d+)*(?:\.\d+)?\b",  # Numbers
            # Years with context
            r"\b(?:in|during|since|before|after|around)\s+\d{4}\b",
            # Dates
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b",
            r"\$\d+(?:,\d+)*(?:\.\d+)?\b",  # Dollar amounts
            # Quantities of people
            r"\b\d+(?:,\d+)*\s+(?:people|users|customers|patients|students)\b",
            # Citations
            r"\b(?:according to|cited by|reported by|study by|research by|survey by)\b",
            # Polish patterns
            r"\b\d+(?:,\d+)*\s+(?:zł|PLN)\b",  # Polish currency
            # Polish months with year
            r"\b(?:styczeń|luty|marzec|kwiecień|maj|czerwiec|lipiec|sierpień|wrzesień|październik|listopad|grudzień)\s+\d{4}\b",
            r"\b(?:w|podczas|od|do|przed|po)\s+\d{4}\b",  # Polish years with context
            # Polish quantities of people
            r"\b\d+(?:,\d+)*\s+(?:osób|ludzi|pracowników|studentów|pacjentów)\b",
            r"\b(?:według|zgodnie z|jak podaje|badania|raport)\b",  # Polish citations
        ]
        self.fact_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.fact_patterns
        ]

        # Common stop words in multiple languages (English, Polish, etc.)
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
            "dlaczego",
            "dlatego",
            "aby",
            "żeby",
            "jeśli",
            "jeżeli",
            "gdyby",
            "niż",
            "niżeli",
            "ani",
            "albo",
            "lecz",
            "jednak",
            "natomiast",
            "zaś",
            "zatem",
            "więc",
            "toteż",
            "dlatego",
            "stąd",
            "potem",
            "następnie",
        }

    def analyze(self, content: str) -> Dict[str, Any]:
        """
        Analyze the information density of the given content.

        Args:
            content: The text content to analyze

        Returns:
            Dictionary containing information density metrics and normalized score
        """
        if not content or len(content.strip()) < 100:
            return {
                "lexical_diversity": 0,
                "fact_density": 0,
                "concept_density": 0,
                "key_concepts": [],
                "normalized_score": 5.0,  # Default middle score for insufficient content
            }

        # Detect if content is likely Polish
        is_polish = self._is_likely_polish(content)

        # Tokenize content with language-aware tokenization
        try:
            # Use simple regex-based tokenization instead of NLTK
            sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]

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
            sentences = [s.strip() for s in re.split(r"[.!?]+", content) if s.strip()]
            words = [
                w.lower() for w in re.findall(r"\b\w+\b", content)
            ]  # \w matches any word character

        # Filter out punctuation, numbers, and stop words
        filtered_words = [
            word
            for word in words
            if word.isalpha()  # Only alphabetic words (no numbers or punctuation)
            and word not in self.stop_words
            and len(word) > 2  # Filter out very short words
        ]

        # Calculate lexical diversity
        lexical_diversity = self._calculate_lexical_diversity(filtered_words)

        # Calculate fact density
        fact_density = self._calculate_fact_density(sentences)

        # Calculate concept density and extract key concepts
        concept_density, key_concepts = self._calculate_concept_density(filtered_words)

        # Calculate normalized score (1-10)
        normalized_score = self._calculate_normalized_score(
            lexical_diversity, fact_density, concept_density
        )

        return {
            "lexical_diversity": round(lexical_diversity, 3),
            "fact_density": round(fact_density, 3),
            "concept_density": round(concept_density, 3),
            "key_concepts": key_concepts[:10],  # Return top 10 concepts
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

    def _calculate_lexical_diversity(self, words: List[str]) -> float:
        """
        Calculate lexical diversity as the ratio of unique words to total words.

        Args:
            words: List of words in the content

        Returns:
            Lexical diversity score (0-1)
        """
        if not words:
            return 0

        unique_words = set(words)
        return len(unique_words) / len(words)

    def _calculate_fact_density(self, sentences: List[str]) -> float:
        """
        Calculate fact density as the proportion of sentences containing factual elements.

        Args:
            sentences: List of sentences in the content

        Returns:
            Fact density score (0-1)
        """
        if not sentences:
            return 0

        factual_sentences = 0

        for sentence in sentences:
            # Check if sentence contains any factual patterns
            is_factual = any(pattern.search(sentence) for pattern in self.fact_patterns)
            if is_factual:
                factual_sentences += 1

        return factual_sentences / len(sentences)

    def _calculate_concept_density(self, words: List[str]) -> Tuple[float, List[str]]:
        """
        Calculate concept density and identify key concepts.

        Args:
            words: List of words in the content

        Returns:
            Tuple of (concept density score (0-1), list of key concepts)
        """
        if not words:
            return 0, []

        # Count word frequencies
        word_counts = Counter(words)

        # Filter out words that appear only once
        significant_words = {
            word: count for word, count in word_counts.items() if count > 1
        }

        # Calculate concept density
        if not words:
            return 0, []

        concept_density = len(significant_words) / len(set(words))

        # Extract key concepts (words with highest frequency)
        key_concepts = [word for word, _ in word_counts.most_common(10)]

        return min(concept_density, 1.0), key_concepts

    def _calculate_normalized_score(
        self, lexical_diversity: float, fact_density: float, concept_density: float
    ) -> float:
        """
        Calculate a normalized information density score on a scale of 1-10.

        Args:
            lexical_diversity: Lexical diversity score (0-1)
            fact_density: Fact density score (0-1)
            concept_density: Concept density score (0-1)

        Returns:
            Normalized score from 1-10
        """
        # Weight the components
        weighted_score = (
            lexical_diversity * 0.4  # Lexical diversity is most important
            + fact_density * 0.4  # Facts are equally important
            + concept_density * 0.2  # Concept density is less important
        )

        # Map to 1-10 scale (higher is better for information density)
        return 1 + weighted_score * 9  # Scale from 1-10
