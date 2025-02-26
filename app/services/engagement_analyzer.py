import re
import math
from typing import Dict, Any, List, Optional, Union
import logging
import string
from collections import Counter
import langdetect  # New import for language detection

logger = logging.getLogger(__name__)


class EngagementAnalyzer:
    """
    Analyzes the engagement potential of article content based on:
    - Emotional content: Presence of emotionally charged language
    - Narrative structure: Storytelling elements and narrative flow
    - Visual elements: Mentions of images, videos, charts, etc.
    - Interactive elements: Calls to action, questions, etc.

    Supports both English and Polish content.
    """

    def __init__(self):
        """Initialize the engagement analyzer with multilingual support."""
        # English patterns for identifying emotional content
        self.emotional_patterns_en = {
            # Positive emotions - English
            "positive": [
                r"\b(?:happy|happiness|joy|joyful|excited|excitement|thrilled|delighted|pleased|glad|satisfied|proud|pride|love|admire|admiration|hope|hopeful|optimistic|optimism|grateful|gratitude|thankful|appreciate|appreciation|inspired|inspiring|inspiration|amazed|amazing|wonderful|excellent|fantastic|great|good|positive|success|successful|achievement|accomplish|accomplished|win|winning|victory|victorious|triumph|triumphant|celebrate|celebration|enjoy|enjoyment|pleasant|pleasing|pleased|pleasure|content|contented|contentment|calm|peaceful|peace|serene|serenity|relaxed|relaxing|comfort|comfortable|confident|confidence)\b",
            ],
            # Negative emotions - English
            "negative": [
                r"\b(?:sad|sadness|unhappy|depressed|depression|upset|angry|anger|furious|fury|outraged|outrage|frustrated|frustration|annoyed|annoying|irritated|irritating|disappointed|disappointment|worried|worry|anxious|anxiety|afraid|fear|scared|terrified|terror|horrified|horror|dread|panic|stressed|stress|overwhelmed|exhausted|exhaustion|tired|fatigue|hurt|painful|pain|suffer|suffering|grief|grieving|mourn|mourning|regret|regretful|sorry|apologize|apology|ashamed|shame|embarrassed|embarrassment|guilty|guilt|jealous|jealousy|envious|envy|hate|hatred|dislike|disgusted|disgust|offended|offense|threatened|threat|confused|confusion|uncertain|uncertainty|doubt|doubtful|skeptical|skepticism|suspicious|suspicion|distrust|distrustful|lonely|loneliness|isolated|isolation|abandoned|rejection|rejected|betrayed|betrayal|desperate|desperation|hopeless|hopelessness|pessimistic|pessimism|disappointed|disappointment|frustrated|frustration)\b",
            ],
            # Surprise/curiosity - English
            "surprise": [
                r"\b(?:surprised|surprise|surprising|shocked|shock|shocking|astonished|astonishment|astonishing|amazed|amazing|amazement|stunned|stunning|startled|startling|unexpected|unanticipated|unforeseen|curious|curiosity|intrigued|intriguing|fascinated|fascinating|fascination|wonder|wonderful|wondering|mysterious|mystery|puzzled|puzzling|puzzle|perplexed|perplexing|bewildered|bewildering|confused|confusing|baffled|baffling)\b",
            ],
        }

        # Polish patterns for identifying emotional content
        self.emotional_patterns_pl = {
            # Positive emotions - Polish
            "positive": [
                r"\b(?:szczęśliwy|szczęśliwa|szczęśliwe|szczęście|radość|radosny|radosna|radosne|podekscytowany|podekscytowana|podekscytowane|podekscytowanie|zachwycony|zachwycona|zachwycone|zachwyt|zadowolony|zadowolona|zadowolone|zadowolenie|dumny|dumna|dumne|duma|miłość|kochać|podziw|podziwiać|nadzieja|pełen nadziei|pełna nadziei|optymistyczny|optymistyczna|optymistyczne|optymizm|wdzięczny|wdzięczna|wdzięczne|wdzięczność|dziękować|doceniać|doceniam|zainspirowany|zainspirowana|zainspirowane|inspiracja|inspirujący|inspirująca|inspirujące|zdumiony|zdumiona|zdumione|zdumiewający|zdumiewająca|zdumiewające|wspaniały|wspaniała|wspaniałe|doskonały|doskonała|doskonałe|fantastyczny|fantastyczna|fantastyczne|świetny|świetna|świetne|dobry|dobra|dobre|pozytywny|pozytywna|pozytywne|sukces|udany|udana|udane|osiągnięcie|osiągać|wygrać|zwycięstwo|zwycięski|zwycięska|zwycięskie|triumf|triumfalny|triumfalna|triumfalne|świętować|świętowanie|cieszyć się|przyjemny|przyjemna|przyjemne|przyjemność|spokojny|spokojna|spokojne|spokój|zrelaksowany|zrelaksowana|zrelaksowane|komfort|komfortowy|komfortowa|komfortowe|pewny|pewna|pewne|pewność)\b",
            ],
            # Negative emotions - Polish
            "negative": [
                r"\b(?:smutny|smutna|smutne|smutek|nieszczęśliwy|nieszczęśliwa|nieszczęśliwe|przygnębiony|przygnębiona|przygnębione|depresja|zdenerwowany|zdenerwowana|zdenerwowane|zły|zła|złe|złość|wściekły|wściekła|wściekłe|wściekłość|oburzony|oburzona|oburzone|oburzenie|sfrustrowany|sfrustrowana|sfrustrowane|frustracja|zirytowany|zirytowana|zirytowane|irytacja|rozczarowany|rozczarowana|rozczarowane|rozczarowanie|zmartwiony|zmartwiona|zmartwione|zmartwienie|niespokojny|niespokojna|niespokojne|niepokój|przestraszony|przestraszona|przestraszone|strach|przerażony|przerażona|przerażone|przerażenie|zgroza|panika|zestresowany|zestresowana|zestresowane|stres|przytłoczony|przytłoczona|przytłoczone|wyczerpany|wyczerpana|wyczerpane|wyczerpanie|zmęczony|zmęczona|zmęczone|zmęczenie|zraniony|zraniona|zranione|ból|bolesny|bolesna|bolesne|cierpieć|cierpienie|żal|żałować|przepraszać|przeprosiny|zawstydzony|zawstydzona|zawstydzone|wstyd|zażenowany|zażenowana|zażenowane|zażenowanie|winny|winna|winne|wina|zazdrosny|zazdrosna|zazdrosne|zazdrość|nienawidzić|nienawiść|nie lubić|obrzydzony|obrzydzona|obrzydzone|obrzydzenie|urażony|urażona|urażone|uraza|zagrożony|zagrożona|zagrożone|zagrożenie|zdezorientowany|zdezorientowana|zdezorientowane|dezorientacja|niepewny|niepewna|niepewne|niepewność|wątpliwość|sceptyczny|sceptyczna|sceptyczne|sceptycyzm|podejrzliwy|podejrzliwa|podejrzliwe|podejrzenie|nieufny|nieufna|nieufne|nieufność|samotny|samotna|samotne|samotność|izolowany|izolowana|izolowane|izolacja|porzucony|porzucona|porzucone|odrzucenie|odrzucony|odrzucona|odrzucone|zdradzony|zdradzona|zdradzone|zdrada|zdesperowany|zdesperowana|zdesperowane|desperacja|beznadziejny|beznadziejna|beznadziejne|beznadziejność|pesymistyczny|pesymistyczna|pesymistyczne|pesymizm)\b",
            ],
            # Surprise/curiosity - Polish
            "surprise": [
                r"\b(?:zaskoczony|zaskoczona|zaskoczone|zaskoczenie|zszokowany|zszokowana|zszokowane|szok|zdumiony|zdumiona|zdumione|zdumienie|osłupiały|osłupiała|osłupiałe|osłupienie|oszołomiony|oszołomiona|oszołomione|nieoczekiwany|nieoczekiwana|nieoczekiwane|nieprzewidziany|nieprzewidziana|nieprzewidziane|ciekawy|ciekawa|ciekawe|ciekawość|zaintrygowany|zaintrygowana|zaintrygowane|zafascynowany|zafascynowana|zafascynowane|fascynacja|zastanawiać się|cudowny|cudowna|cudowne|cud|tajemniczy|tajemnicza|tajemnicze|tajemnica|zagadkowy|zagadkowa|zagadkowe|zagadka|zdezorientowany|zdezorientowana|zdezorientowane|zagubiony|zagubiona|zagubione)\b",
            ],
        }

        # Compile emotional patterns for both languages
        for category, patterns in self.emotional_patterns_en.items():
            self.emotional_patterns_en[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        for category, patterns in self.emotional_patterns_pl.items():
            self.emotional_patterns_pl[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # English patterns for narrative elements
        self.narrative_patterns_en = [
            r"\b(?:story|stories|narrative|account|chronicle|tale|anecdote|experience|journey|adventure|episode|incident|event|scenario|situation|case|example|illustration)\b",
            r"\b(?:first|initially|originally|at first|to begin with|starting|started|began|beginning|once|earlier|previously|before|prior to)\b",
            r"\b(?:then|next|after that|subsequently|following this|afterward|afterwards|later|soon after|eventually|finally|lastly|ultimately|in the end|at last)\b",
            r"\b(?:because|since|as|due to|owing to|thanks to|result of|consequently|therefore|thus|hence|so|accordingly|as a result)\b",
            r"\b(?:however|but|yet|nevertheless|nonetheless|although|though|even though|despite|in spite of|regardless|notwithstanding|on the other hand|conversely|instead|rather|alternatively)\b",
        ]

        # Polish patterns for narrative elements
        self.narrative_patterns_pl = [
            r"\b(?:historia|historie|opowieść|opowieści|narracja|relacja|kronika|opowiadanie|anegdota|doświadczenie|podróż|przygoda|epizod|incydent|wydarzenie|scenariusz|sytuacja|przypadek|przykład|ilustracja)\b",
            r"\b(?:najpierw|początkowo|pierwotnie|na początku|zaczynając|zaczął|zaczęła|zaczęło|zaczynając|rozpoczął|rozpoczęła|rozpoczęło|kiedyś|wcześniej|poprzednio|przedtem|przed)\b",
            r"\b(?:potem|następnie|po tym|później|wkrótce potem|ostatecznie|w końcu|na koniec|wreszcie)\b",
            r"\b(?:ponieważ|gdyż|bo|z powodu|dzięki|w wyniku|w rezultacie|w konsekwencji|dlatego|zatem|więc|tak więc|w związku z tym)\b",
            r"\b(?:jednak|ale|lecz|niemniej|mimo to|chociaż|choć|pomimo|mimo|niezależnie od|z drugiej strony|odwrotnie|zamiast|raczej|alternatywnie)\b",
        ]

        # Compile narrative patterns for both languages
        self.narrative_patterns_en = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.narrative_patterns_en
        ]

        self.narrative_patterns_pl = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.narrative_patterns_pl
        ]

        # Similar pattern definitions for visual_patterns and interactive_patterns
        # (I'm omitting the full Polish translations for brevity, but they would follow the same pattern)

        # English patterns for visual elements
        self.visual_patterns_en = [
            r"\b(?:image|images|picture|pictures|photo|photos|photograph|photographs|illustration|illustrations|figure|figures|diagram|diagrams|chart|charts|graph|graphs|infographic|infographics|map|maps|screenshot|screenshots|graphic|graphics|drawing|drawings|sketch|sketches|painting|paintings|portrait|portraits|landscape|landscapes|scene|scenes|view|views|visual|visuals|visualization|visualizations)\b",
            # ... other English visual patterns
        ]

        # Polish patterns for visual elements
        self.visual_patterns_pl = [
            r"\b(?:obraz|obrazy|obrazek|obrazki|zdjęcie|zdjęcia|fotografia|fotografie|ilustracja|ilustracje|figura|figury|diagram|diagramy|wykres|wykresy|infografika|infografiki|mapa|mapy|zrzut ekranu|zrzuty ekranu|grafika|grafiki|rysunek|rysunki|szkic|szkice|obraz|obrazy|portret|portrety|krajobraz|krajobrazy|scena|sceny|widok|widoki|wizualizacja|wizualizacje)\b",
            # ... other Polish visual patterns
        ]

        # Compile visual patterns for both languages
        self.visual_patterns_en = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.visual_patterns_en
        ]

        self.visual_patterns_pl = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.visual_patterns_pl
        ]

        # English patterns for interactive elements
        self.interactive_patterns_en = [
            r"\b(?:click|tap|swipe|scroll|drag|drop|select|choose|pick|check|uncheck|mark|toggle|switch|press|push|pull|slide|move|navigate|browse|search|find|locate|access|enter|input|type|write|edit|modify|update|change|adjust|customize|customise|personalize|personalise|configure|set up|install|download|upload|share|send|submit|post|publish|comment|reply|respond|feedback|contact|reach out|call|email|message|chat|discuss|talk|communicate|connect|follow|subscribe|sign up|register|join|participate|engage|interact|try|test|experiment|explore|discover|learn|read|study|practice|exercise|play|use|utilize|apply|implement|execute|perform|complete|finish|continue|proceed|go|start|begin|initiate|launch|activate|enable|disable|turn on|turn off)\b",
            # ... other English interactive patterns
        ]

        # Polish patterns for interactive elements
        self.interactive_patterns_pl = [
            r"\b(?:kliknij|dotknij|przesuń|przewiń|przeciągnij|upuść|wybierz|zaznacz|odznacz|oznacz|przełącz|naciśnij|pociągnij|przesuń|porusz|nawiguj|przeglądaj|szukaj|znajdź|zlokalizuj|uzyskaj dostęp|wprowadź|wpisz|napisz|edytuj|modyfikuj|aktualizuj|zmień|dostosuj|spersonalizuj|konfiguruj|ustaw|zainstaluj|pobierz|wyślij|udostępnij|wyślij|prześlij|opublikuj|skomentuj|odpowiedz|zareaguj|skontaktuj się|zadzwoń|napisz|porozmawiaj|komunikuj się|połącz|śledź|subskrybuj|zarejestruj się|dołącz|uczestniczyć|zaangażuj się|wypróbuj|przetestuj|eksperymentuj|odkryj|ucz się|czytaj|studiuj|ćwicz|graj|użyj|zastosuj|wdrażaj|wykonaj|ukończ|zakończ|kontynuuj|idź|rozpocznij|zainicjuj|uruchom|aktywuj|włącz|wyłącz)\b",
            # ... other Polish interactive patterns
        ]

        # Compile interactive patterns for both languages
        self.interactive_patterns_en = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.interactive_patterns_en
        ]

        self.interactive_patterns_pl = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.interactive_patterns_pl
        ]

    def analyze(self, content: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the engagement potential of the given content.

        Args:
            content: The text content to analyze
            title: The article title (optional)

        Returns:
            Dictionary containing engagement metrics and normalized score
        """
        if not content or len(content.strip()) < 100:
            return {
                "emotional_score": 0,
                "narrative_score": 0,
                "visual_score": 0,
                "interactive_score": 0,
                "emotion_counts": {"positive": 0, "negative": 0, "surprise": 0},
                "normalized_score": 5.0,  # Default middle score for insufficient content
                "language": "unknown",
            }

        # Detect language
        try:
            language = langdetect.detect(content)
        except BaseException:
            # Default to English if detection fails
            language = "en"

        # Combine title and content if title is provided
        full_text = f"{title}. {content}" if title else content

        # Calculate emotional content score
        emotion_counts, emotional_score = self._calculate_emotional_score(
            full_text, language
        )

        # Calculate narrative structure score
        narrative_score = self._calculate_narrative_score(content, language)

        # Calculate visual elements score
        visual_score = self._calculate_visual_score(content, language)

        # Calculate interactive elements score
        interactive_score = self._calculate_interactive_score(content, language)

        # Calculate normalized score (1-10)
        normalized_score = self._calculate_normalized_score(
            emotional_score, narrative_score, visual_score, interactive_score
        )

        return {
            "emotional_score": round(emotional_score, 3),
            "narrative_score": round(narrative_score, 3),
            "visual_score": round(visual_score, 3),
            "interactive_score": round(interactive_score, 3),
            "emotion_counts": emotion_counts,
            "normalized_score": round(normalized_score, 2),
            "language": language,
        }

    def _calculate_emotional_score(self, content: str, language: str = "en") -> tuple:
        """
        Calculate emotional content score based on emotional language.

        Args:
            content: The text content to analyze
            language: The detected language code

        Returns:
            Tuple of (emotion_counts, emotional_score)
        """
        # Count emotional terms by category
        emotion_counts = {"positive": 0, "negative": 0, "surprise": 0}

        # Calculate total word count for normalization
        words = re.findall(r"\b\w+\b", content.lower())
        total_words = len(words)

        if total_words == 0:
            return emotion_counts, 0

        # Select appropriate patterns based on language
        if language == "pl":
            patterns = self.emotional_patterns_pl
        else:
            patterns = self.emotional_patterns_en

        # Count emotional terms
        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                matches = pattern.findall(content)
                emotion_counts[category] += len(matches)

        # Calculate total emotional terms
        total_emotional = sum(emotion_counts.values())

        # Calculate emotional density (emotional terms per 100 words)
        emotional_density = (total_emotional / total_words) * 100

        # Calculate emotional diversity (distribution across categories)
        if total_emotional > 0:
            category_ratios = [
                count / total_emotional
                for count in emotion_counts.values()
                if count > 0
            ]
            emotional_diversity = len(
                [r for r in category_ratios if r >= 0.1]
            )  # Categories with at least 10% representation
        else:
            emotional_diversity = 0

        # Combine density and diversity for final score
        # Scale density to 0-1 range (assuming max density around 10%)
        scaled_density = min(1.0, emotional_density / 10)

        # Scale diversity to 0-1 range (max diversity is 3 categories)
        scaled_diversity = emotional_diversity / 3

        # Weighted combination (density is more important)
        emotional_score = (scaled_density * 0.7) + (scaled_diversity * 0.3)

        return emotion_counts, emotional_score

    def _calculate_narrative_score(self, content: str, language: str = "en") -> float:
        """
        Calculate narrative structure score based on storytelling elements.

        Args:
            content: The text content to analyze
            language: The detected language code

        Returns:
            Narrative structure score (0-1)
        """
        # Select appropriate patterns based on language
        if language == "pl":
            patterns = self.narrative_patterns_pl
        else:
            patterns = self.narrative_patterns_en

        # Count narrative elements
        narrative_count = 0
        for pattern in patterns:
            matches = pattern.findall(content)
            narrative_count += len(matches)

        # Calculate total word count for normalization
        words = re.findall(r"\b\w+\b", content.lower())
        total_words = len(words)

        if total_words == 0:
            return 0

        # Calculate narrative density (narrative elements per 100 words)
        narrative_density = (narrative_count / total_words) * 100

        # Scale to 0-1 range (assuming max density around 5%)
        narrative_score = min(1.0, narrative_density / 5)

        return narrative_score

    def _calculate_visual_score(self, content: str, language: str = "en") -> float:
        """
        Calculate visual elements score based on mentions of visual content.

        Args:
            content: The text content to analyze
            language: The detected language code

        Returns:
            Visual elements score (0-1)
        """
        # Select appropriate patterns based on language
        if language == "pl":
            patterns = self.visual_patterns_pl
        else:
            patterns = self.visual_patterns_en

        # Count visual elements
        visual_count = 0
        for pattern in patterns:
            matches = pattern.findall(content)
            visual_count += len(matches)

        # Calculate total word count for normalization
        words = re.findall(r"\b\w+\b", content.lower())
        total_words = len(words)

        if total_words == 0:
            return 0

        # Calculate visual density (visual elements per 100 words)
        visual_density = (visual_count / total_words) * 100

        # Scale to 0-1 range (assuming max density around 3%)
        visual_score = min(1.0, visual_density / 3)

        return visual_score

    def _calculate_interactive_score(self, content: str, language: str = "en") -> float:
        """
        Calculate interactive elements score based on calls to action, questions, etc.

        Args:
            content: The text content to analyze
            language: The detected language code

        Returns:
            Interactive elements score (0-1)
        """
        # Select appropriate patterns based on language
        if language == "pl":
            patterns = self.interactive_patterns_pl
        else:
            patterns = self.interactive_patterns_en

        # Count interactive elements
        interactive_count = 0
        for pattern in patterns:
            matches = pattern.findall(content)
            interactive_count += len(matches)

        # Calculate total word count for normalization
        words = re.findall(r"\b\w+\b", content.lower())
        total_words = len(words)

        if total_words == 0:
            return 0

        # Calculate interactive density (interactive elements per 100 words)
        interactive_density = (interactive_count / total_words) * 100

        # Scale to 0-1 range (assuming max density around 5%)
        interactive_score = min(1.0, interactive_density / 5)

        return interactive_score

    def _calculate_normalized_score(
        self,
        emotional_score: float,
        narrative_score: float,
        visual_score: float,
        interactive_score: float,
    ) -> float:
        """
        Calculate a normalized engagement potential score on a scale of 1-10.

        Args:
            emotional_score: Emotional content score (0-1)
            narrative_score: Narrative structure score (0-1)
            visual_score: Visual elements score (0-1)
            interactive_score: Interactive elements score (0-1)

        Returns:
            Normalized score from 1-10
        """
        # Weight the components
        weighted_score = (
            emotional_score * 0.35  # Emotional content is most important
            + narrative_score * 0.25  # Narrative structure is important
            + visual_score * 0.2  # Visual elements are moderately important
            + interactive_score * 0.2  # Interactive elements are moderately important
        )

        # Map to 1-10 scale (higher is better for engagement potential)
        return 1 + weighted_score * 9  # Scale from 1-10
