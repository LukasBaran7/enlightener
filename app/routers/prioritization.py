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

        # Define component weights according to the spec
        self.component_weights = {
            "quality": 0.25,
            "information_density": 0.15,
            "readability": 0.15,
            "topic_relevance": 0.20,
            "freshness": 0.10,
            "engagement_potential": 0.15,
        }

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

    async def calculate_priority_scores(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate the final priority score for each article based on component scores.

        Args:
            articles: List of article documents with all component metrics

        Returns:
            List of articles with priority scores
        """
        for article in articles:
            # Initialize component scores dictionary
            component_scores = {}

            # Get quality score (placeholder - to be implemented)
            # For now, use a default value of 7.0
            component_scores["quality"] = 7.0

            # Get information density score
            if "information_density" in article:
                component_scores["information_density"] = article[
                    "information_density"
                ].get("normalized_score", 5.0)
            else:
                component_scores["information_density"] = 5.0

            # Get readability score
            if "readability" in article:
                component_scores["readability"] = article["readability"].get(
                    "normalized_score", 5.0
                )
            else:
                component_scores["readability"] = 5.0

            # Get topic relevance score
            if "topic_relevance" in article:
                component_scores["topic_relevance"] = article["topic_relevance"].get(
                    "normalized_score", 5.0
                )
            else:
                component_scores["topic_relevance"] = 5.0

            # Get freshness score
            if "freshness" in article:
                component_scores["freshness"] = article["freshness"].get(
                    "normalized_score", 5.0
                )
            else:
                component_scores["freshness"] = 5.0

            # Get engagement potential score
            if "engagement_potential" in article:
                component_scores["engagement_potential"] = article[
                    "engagement_potential"
                ].get("normalized_score", 5.0)
            else:
                component_scores["engagement_potential"] = 5.0

            # Calculate weighted sum according to the formula in the spec
            priority_score = (
                sum(
                    score * self.component_weights[component]
                    for component, score in component_scores.items()
                )
                * 10
            )  # Scale to 0-100

            # Add priority score and component scores to article
            article["priority_score"] = round(priority_score, 1)
            article["component_scores"] = component_scores

        return articles

    async def format_prioritized_articles(
        self, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format the prioritized articles according to the output format in the spec.

        Args:
            articles: List of article documents with priority scores

        Returns:
            List of formatted article results
        """
        formatted_articles = []

        for article in articles:
            # Convert ObjectId to string if present
            article_id = str(article.get("_id", "")) if article.get("_id") else ""

            # Ensure tags is always a dictionary
            tags = article.get("tags", {})
            if tags is None:
                tags = {}

            formatted_article = {
                # Original MongoDB fields
                "_id": article_id,
                "id": article.get("id", ""),
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "author": article.get("author", ""),
                "source": article.get("source", ""),
                "category": article.get("category", ""),
                "location": article.get("location", ""),
                "tags": tags,  # Use the validated tags value
                "site_name": article.get("site_name", ""),
                "word_count": article.get("word_count", 0),
                "created_at": article.get("created_at", None),
                "updated_at": article.get("updated_at", None),
                "published_date": article.get("published_date", 0),
                "summary": article.get("summary", ""),
                "image_url": article.get("image_url", ""),
                "content": article.get("content", ""),
                "source_url": article.get("source_url", ""),
                "notes": article.get("notes", ""),
                "parent_id": article.get("parent_id", None),
                "reading_progress": article.get("reading_progress", 0),
                "first_opened_at": article.get("first_opened_at", None),
                "last_opened_at": article.get("last_opened_at", None),
                "saved_at": article.get("saved_at", None),
                "last_moved_at": article.get("last_moved_at", None),
                # Prioritization fields
                "article_id": article_id,  # Duplicate for backward compatibility
                "priority_score": article.get("priority_score", 0),
                "component_scores": article.get("component_scores", {}),
            }

            formatted_articles.append(formatted_article)

        # Sort articles by priority score in descending order
        formatted_articles.sort(key=lambda x: x["priority_score"], reverse=True)

        return formatted_articles


@router.get("/sample", response_model=Dict[str, Any])
async def get_prioritization_sample(
    limit: int = Query(
        default=10, ge=1, le=20, description="Number of articles to return"
    ),
    sample_size: int = Query(
        default=25,
        ge=10,
        le=200,
        description="Number of articles to sample and process",
    ),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Get a sample of articles with prioritization scores.

    Processes a larger sample (default 100) of articles and returns the top N (default 10)
    with highest priority scores. Also includes metadata about min/max scores.
    """
    try:
        service = PrioritizationService(db)

        # Get random articles (larger sample)
        articles = await service.get_random_articles_for_prioritization(sample_size)

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

        # Calculate priority scores
        prioritized_articles = await service.calculate_priority_scores(
            analyzed_articles
        )

        # Format articles for output
        formatted_articles = await service.format_prioritized_articles(
            prioritized_articles
        )

        # Get min and max scores from all processed articles
        all_scores = [article["priority_score"] for article in formatted_articles]
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0

        # Take only the top N articles
        top_articles = formatted_articles[:limit]

        # Add metadata about scores
        response = {
            "articles": top_articles,
            "metadata": {
                "total_processed": len(formatted_articles),
                "min_score": min_score,
                "max_score": max_score,
                "returned_count": len(top_articles),
            },
        }

        return response

    except Exception as e:
        logger.error(f"Error in get_prioritization_sample: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving prioritization sample: {str(e)}"
        )
