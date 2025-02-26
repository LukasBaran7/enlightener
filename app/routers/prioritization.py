from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database
from app.services.content_extractor import ContentExtractor
from app.services.readability_analyzer import ReadabilityAnalyzer
from app.services.information_density_analyzer import InformationDensityAnalyzer
from app.services.topic_relevance_analyzer import TopicRelevanceAnalyzer
from app.services.freshness_analyzer import FreshnessAnalyzer
from app.services.engagement_analyzer import EngagementAnalyzer
from bson import ObjectId
from datetime import datetime
import logging

router = APIRouter(prefix="/prioritization", tags=["prioritization"])
logger = logging.getLogger(__name__)


class PrioritizationService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.content_extractor = ContentExtractor()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.information_density_analyzer = InformationDensityAnalyzer()
        self.topic_relevance_analyzer = TopicRelevanceAnalyzer()
        self.freshness_analyzer = FreshnessAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()

    async def get_random_articles_for_prioritization(
        self, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve random articles from the 'later' collection for prioritization.

        Args:
            limit: Maximum number of articles to retrieve

        Returns:
            List of article documents
        """
        pipeline = [
            {"$match": {"content": {"$exists": True}}},  # Ensure content field exists
            {"$sample": {"size": limit}},  # Get random articles
        ]

        cursor = self.db.later.aggregate(pipeline)
        articles = await cursor.to_list(length=None)

        return articles

    async def extract_content_for_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract content for a list of articles.

        Args:
            articles: List of article documents

        Returns:
            List of articles with extracted content
        """
        result = []

        for article_doc in articles:
            try:
                # Convert ObjectId to string before validation
                article_copy = article_doc.copy()
                if "_id" in article_copy and isinstance(article_copy["_id"], ObjectId):
                    article_copy["_id"] = str(article_copy["_id"])

                # Convert MongoDB document to Article model
                article = Article.model_validate(article_copy)

                # Extract content
                extracted_content = await self.content_extractor.extract_content(
                    article
                )

                # Add extracted content to the article document
                article_doc["extracted_content"] = extracted_content

                # Add to result if content was successfully extracted
                if extracted_content:
                    result.append(article_doc)

            except Exception as e:
                logger.error(
                    f"Error processing article {article_doc.get('_id')}: {str(e)}"
                )

        return result

    async def analyze_readability(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze readability for a list of articles with extracted content.

        Args:
            articles: List of article documents with extracted content

        Returns:
            List of articles with readability metrics
        """
        for article in articles:
            content = article.get("extracted_content", "")
            if content:
                # Analyze readability
                readability_metrics = self.readability_analyzer.analyze(content)

                # Add readability metrics to article
                article["readability"] = readability_metrics
            else:
                # Default metrics for articles without content
                article["readability"] = {
                    "flesch_reading_ease": 0,
                    "smog_index": 0,
                    "coleman_liau_index": 0,
                    "automated_readability_index": 0,
                    "complexity_level": "Unknown",
                    "normalized_score": 5.0,
                }

        return articles

    async def analyze_information_density(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze information density for a list of articles with extracted content.

        Args:
            articles: List of article documents with extracted content

        Returns:
            List of articles with information density metrics
        """
        for article in articles:
            content = article.get("extracted_content", "")
            if content:
                # Analyze information density
                density_metrics = self.information_density_analyzer.analyze(content)

                # Add information density metrics to article
                article["information_density"] = density_metrics
            else:
                # Default metrics for articles without content
                article["information_density"] = {
                    "lexical_diversity": 0,
                    "fact_density": 0,
                    "concept_density": 0,
                    "key_concepts": [],
                    "normalized_score": 5.0,
                }

        return articles

    async def analyze_topic_relevance(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze topic relevance for a list of articles with extracted content.

        Args:
            articles: List of article documents with extracted content

        Returns:
            List of articles with topic relevance metrics
        """
        for article in articles:
            content = article.get("extracted_content", "")
            if content:
                # Analyze topic relevance
                topic_relevance_metrics = self.topic_relevance_analyzer.analyze(content)

                # Add topic relevance metrics to article
                article["topic_relevance"] = topic_relevance_metrics
            else:
                # Default metrics for articles without content
                article["topic_relevance"] = {
                    "top_topics": [],
                    "topic_matches": {},
                    "normalized_score": 5.0,
                }

        return articles

    async def analyze_freshness(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze content freshness for a list of articles.

        Args:
            articles: List of article documents with extracted content

        Returns:
            List of articles with freshness metrics
        """
        for article in articles:
            content = article.get("extracted_content", "")
            if content:
                # Get publication date if available
                published_date = None
                if "published_date" in article and article["published_date"]:
                    try:
                        # Handle millisecond timestamp
                        if isinstance(article["published_date"], int):
                            published_date = datetime.fromtimestamp(
                                article["published_date"] / 1000
                            )
                        # Handle datetime object
                        elif isinstance(article["published_date"], datetime):
                            published_date = article["published_date"]
                    except Exception as e:
                        logger.warning(f"Error parsing published_date: {str(e)}")

                # Determine category based on article metadata or topic relevance
                category = "default"
                if "category" in article and article["category"]:
                    category = article["category"]
                elif "topic_relevance" in article and article["topic_relevance"].get(
                    "top_topics"
                ):
                    # Map top topic to a category if possible
                    top_topic = (
                        article["topic_relevance"]["top_topics"][0]
                        if article["topic_relevance"]["top_topics"]
                        else None
                    )
                    if top_topic == "technology":
                        category = "technology"
                    elif top_topic == "science":
                        category = "science"
                    elif top_topic in ["politics", "business", "finance"]:
                        category = "news"
                    elif top_topic in ["education", "health"]:
                        category = "evergreen"

                # Analyze freshness
                freshness_metrics = self.freshness_analyzer.analyze(
                    content, published_date, category
                )

                # Add freshness metrics to article
                article["freshness"] = freshness_metrics
            else:
                # Default metrics for articles without content
                article["freshness"] = {
                    "age_days": 0,
                    "temporal_references_count": 0,
                    "decay_rate": 180,  # Default decay rate
                    "is_recent": False,
                    "normalized_score": 5.0,
                }

        return articles

    async def analyze_engagement_potential(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze engagement potential for a list of articles.

        Args:
            articles: List of article documents with extracted content

        Returns:
            List of articles with engagement potential metrics
        """
        for article in articles:
            content = article.get("extracted_content", "")
            title = article.get("title", "")

            if content:
                # Analyze engagement potential
                engagement_metrics = self.engagement_analyzer.analyze(content, title)

                # Add engagement metrics to article
                article["engagement_potential"] = engagement_metrics
            else:
                # Default metrics for articles without content
                article["engagement_potential"] = {
                    "emotional_score": 0,
                    "narrative_score": 0,
                    "visual_score": 0,
                    "interactive_score": 0,
                    "emotion_counts": {"positive": 0, "negative": 0, "surprise": 0},
                    "normalized_score": 5.0,
                }

        return articles


@router.get("/sample", response_model=List[Dict[str, Any]])
async def get_prioritization_sample(
    limit: int = Query(
        default=10, ge=1, le=20, description="Number of articles to sample"
    ),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Get a sample of articles with extracted content for prioritization testing.
    """
    try:
        service = PrioritizationService(db)

        # Get random articles
        articles = await service.get_random_articles_for_prioritization(limit)

        # Extract content for articles
        processed_articles = await service.extract_content_for_articles(articles)

        # Analyze readability
        analyzed_articles = await service.analyze_readability(processed_articles)

        # Analyze information density
        analyzed_articles = await service.analyze_information_density(analyzed_articles)

        # Analyze topic relevance
        analyzed_articles = await service.analyze_topic_relevance(analyzed_articles)

        # Analyze freshness
        analyzed_articles = await service.analyze_freshness(analyzed_articles)

        # Analyze engagement potential
        analyzed_articles = await service.analyze_engagement_potential(
            analyzed_articles
        )

        # Convert ObjectId to string for JSON serialization
        for article in analyzed_articles:
            if "_id" in article:
                article["_id"] = str(article["_id"])

        return analyzed_articles

    except Exception as e:
        logger.error(f"Error in get_prioritization_sample: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving prioritization sample: {str(e)}"
        )
