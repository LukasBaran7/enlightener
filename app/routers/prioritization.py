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
from datetime import datetime, timezone, timedelta
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

        # Collect all article IDs
        article_ids = []
        article_id_map = {}

        for article_doc in articles:
            article_copy = article_doc.copy()
            if "_id" in article_copy and isinstance(article_copy["_id"], ObjectId):
                article_copy["_id"] = str(article_copy["_id"])

            # Convert MongoDB document to Article model
            try:
                article = Article.model_validate(article_copy)
                if article.id:
                    article_ids.append(article.id)
                    article_id_map[article.id] = article
                    article_doc["article_model"] = article
            except Exception as e:
                logger.error(
                    f"Error validating article {article_copy.get('_id')}: {str(e)}"
                )

        # Batch retrieve HTML content for all articles
        html_docs = {}
        if article_ids:
            cursor = self.db.later_html.find({"article_id": {"$in": article_ids}})
            async for html_doc in cursor:
                if "article_id" in html_doc and "html" in html_doc and html_doc["html"]:
                    html_docs[html_doc["article_id"]] = html_doc["html"]

            logger.info(
                f"Retrieved {len(html_docs)} HTML documents from later_html collection"
            )

        # Process each article with the retrieved HTML content
        for article_doc in articles:
            try:
                if "article_model" not in article_doc:
                    continue

                article = article_doc["article_model"]

                # Assign HTML content if available
                if article.id in html_docs:
                    logger.info(
                        f"Using HTML content from later_html for article {article.id}"
                    )
                    article.html_content = html_docs[article.id]

                # Extract content
                extracted_content = await self.content_extractor.extract_content(
                    article
                )

                # Add extracted content to the article document
                article_doc["extracted_content"] = extracted_content

                # Remove the temporary article_model field
                del article_doc["article_model"]

                # Add to result if content was successfully extracted
                if extracted_content:
                    result.append(article_doc)

            except Exception as e:
                logger.error(
                    f"Error processing article {article_doc.get('_id')}: {str(e)}"
                )
                # Remove the temporary article_model field if it exists
                if "article_model" in article_doc:
                    del article_doc["article_model"]

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

    async def save_prioritization_results(self, articles: List[Dict[str, Any]]) -> None:
        """
        Save prioritization results back to the database.

        Args:
            articles: List of article documents with priority scores
        """
        for article in articles:
            article_id = article.get("_id")
            if not article_id:
                continue

            # Prepare update data - only include prioritization fields
            update_data = {
                "priority_score": article.get("priority_score", 0),
                "component_scores": article.get("component_scores", {}),
                "priority_score_updated_at": datetime.now(timezone.utc),
            }

            # Add analysis results if they exist
            if "readability" in article:
                update_data["readability"] = article["readability"]
            if "information_density" in article:
                update_data["information_density"] = article["information_density"]
            if "topic_relevance" in article:
                update_data["topic_relevance"] = article["topic_relevance"]
            if "freshness" in article:
                update_data["freshness"] = article["freshness"]
            if "engagement_potential" in article:
                update_data["engagement_potential"] = article["engagement_potential"]

            try:
                # Update the article in the database
                await self.db.later.update_one(
                    {"_id": article_id}, {"$set": update_data}
                )
            except Exception as e:
                logger.error(
                    f"Error saving prioritization results for article {article_id}: {str(e)}"
                )

    async def check_existing_scores(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check for existing scores and separate articles that need processing from those that don't.

        Args:
            articles: List of article documents

        Returns:
            Dictionary with 'to_process' and 'already_scored' lists
        """
        to_process = []
        already_scored = []

        for article in articles:
            # Check if article already has priority score
            if "priority_score" in article and article["priority_score"] is not None:
                already_scored.append(article)
            else:
                to_process.append(article)

        return {"to_process": to_process, "already_scored": already_scored}


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
    Get a random sample of articles from the 'later' collection that have scoring data.

    Args:
        limit: Number of articles to return
        sample_size: Number of articles to sample from

    Returns:
        Dictionary with articles and metadata
    """
    try:
        # Query for random articles with scoring data
        pipeline = [
            {
                "$match": {
                    "priority_score": {"$exists": True},
                    "component_scores": {"$exists": True},
                }
            },
            {"$sample": {"size": sample_size}},
            {"$sort": {"priority_score": -1}},  # Sort by priority score (descending)
            {"$limit": limit},
        ]

        cursor = db.later.aggregate(pipeline)
        articles = await cursor.to_list(length=None)

        # Format articles for response
        formatted_articles = []
        for article in articles:
            # Convert ObjectId to string
            if "_id" in article and isinstance(article["_id"], str) is False:
                article["_id"] = str(article["_id"])

            formatted_articles.append(article)

        # Calculate min and max scores for backward compatibility
        all_scores = [
            article.get("priority_score", 0) for article in formatted_articles
        ]
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0

        # Add metadata
        response = {
            "articles": formatted_articles,
            "metadata": {
                "total_processed": sample_size,  # Renamed from total_sampled for backward compatibility
                "min_score": min_score,  # Added for backward compatibility
                "max_score": max_score,  # Added for backward compatibility
                "returned_count": len(formatted_articles),
            },
        }

        return response

    except Exception as e:
        logger.error(f"Error in get_prioritization_sample: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving prioritization sample: {str(e)}"
        )


@router.get("/low-priority", response_model=Dict[str, Any])
async def get_low_priority_articles(
    min_age_days: int = Query(30, description="Minimum age of articles in days"),
    limit: int = Query(10, description="Maximum number of articles to return"),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> Dict[str, Any]:
    """
    Get articles that are candidates for archiving based on low priority.

    This endpoint uses the stored component_scores and other MongoDB fields
    rather than calculating them on the fly.

    Args:
        min_age_days: Minimum age of articles in days to consider
        limit: Maximum number of articles to return

    Returns:
        Dictionary with articles and metadata
    """
    try:
        service = PrioritizationService(db)

        # Calculate cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=min_age_days)
        cutoff_timestamp = int(
            cutoff_date.timestamp() * 1000
        )  # Convert to milliseconds

        # Query for low priority articles
        pipeline = [
            {
                "$match": {
                    # Only include articles with scoring data
                    "priority_score": {"$exists": True},
                    "component_scores": {"$exists": True},
                    "$or": [
                        # Articles with low priority score
                        {"priority_score": {"$lte": 30.0}},
                        # Articles with old publication date
                        {"published_date": {"$lt": cutoff_timestamp}},
                        # Articles with low component scores
                        {"component_scores.readability": {"$lte": 3.0}},
                        {"component_scores.information_density": {"$lte": 3.0}},
                        {"component_scores.topic_relevance": {"$lte": 3.0}},
                        # Articles saved long ago but never opened
                        {
                            "saved_at": {"$lt": cutoff_date},
                            "first_opened_at": {"$exists": False},
                        },
                        # Articles not opened in a long time
                        {
                            "last_opened_at": {
                                "$lt": datetime.now(timezone.utc) - timedelta(days=365)
                            }
                        },
                    ],
                }
            },
            # First get a larger sample of matching articles
            {
                "$sample": {"size": limit * 3}
            },  # Get 3x the requested limit for more variety
            # Sort by priority score (ascending)
            {"$sort": {"priority_score": 1}},
            # Then take a random subset of the top matches
            {"$sample": {"size": limit}},
        ]

        cursor = db.later.aggregate(pipeline)
        articles = await cursor.to_list(length=None)

        # Format articles for response
        formatted_articles = []
        for article in articles:
            # Convert ObjectId to string
            if "_id" in article and isinstance(article["_id"], str) is False:
                article["_id"] = str(article["_id"])

            # Add archive recommendation reason
            reasons = []

            # Check for low priority score
            if article.get("priority_score", 100) <= 30.0:
                reasons.append("low_priority_score")

            # Check for old publication date
            pub_date = article.get("published_date")
            if pub_date and (isinstance(pub_date, int) and pub_date < cutoff_timestamp):
                reasons.append("old_publication_date")

            # Check for failed content extraction
            if article.get("content_extraction_failed") or not article.get("content"):
                reasons.append("content_extraction_failed")

            # Check for low component scores
            component_scores = article.get("component_scores", {})
            if component_scores.get("readability", 10) <= 3.0:
                reasons.append("low_readability")
            if component_scores.get("information_density", 10) <= 3.0:
                reasons.append("low_information_density")
            if component_scores.get("topic_relevance", 10) <= 3.0:
                reasons.append("low_topic_relevance")

            # Check for long-term neglect
            saved_at = article.get("saved_at")
            first_opened_at = article.get("first_opened_at")
            if (
                saved_at
                and not first_opened_at
                and isinstance(saved_at, datetime)
                and (datetime.now(timezone.utc) - saved_at).days > 180
            ):
                reasons.append("long_term_neglect")

            # Check for stale interest
            last_opened_at = article.get("last_opened_at")
            if (
                last_opened_at
                and isinstance(last_opened_at, datetime)
                and (datetime.now(timezone.utc) - last_opened_at).days > 365
            ):
                reasons.append("stale_interest")

            # Check for abandoned reading attempts
            reading_progress = article.get("reading_progress", 1.0)
            if (
                reading_progress < 0.2
                and first_opened_at
                and last_opened_at
                and first_opened_at != last_opened_at
            ):
                reasons.append("abandoned_reading")

            # Check for missing critical metadata
            if (
                not article.get("title")
                or not article.get("author")
                or not article.get("url")
            ):
                reasons.append("missing_critical_metadata")

            # Add reasons to article
            article["archive_reasons"] = reasons
            formatted_articles.append(article)

        # Add metadata about scores
        response = {
            "articles": formatted_articles,
            "metadata": {
                "total_processed": len(formatted_articles),
                "returned_count": len(formatted_articles),
                "criteria": {
                    "min_age_days": min_age_days,
                },
            },
        }

        return response

    except Exception as e:
        logger.error(f"Error in get_low_priority_articles: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving low priority articles: {str(e)}"
        )


@router.get("/llm-sample", response_model=Dict[str, Any])
async def get_llm_scored_sample(
    limit: int = Query(
        default=10, ge=1, le=20, description="Number of articles to return"
    ),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Get a sample of articles sorted by LLM score (highest first).
    """
    try:
        # First get articles that have LLM scores
        pipeline = [
            # Join with llm_scores collection
            {
                "$lookup": {
                    "from": "llm_scores",
                    "localField": "id",
                    "foreignField": "article_id",
                    "as": "llm_score_data",
                }
            },
            # Filter to only include articles with LLM scores
            {"$match": {"llm_score_data": {"$ne": []}}},
            # Add LLM score fields to the article
            {
                "$addFields": {
                    "llm_score": {"$arrayElemAt": ["$llm_score_data.llm_score", 0]},
                    "llm_analysis": {"$arrayElemAt": ["$llm_score_data.analysis", 0]},
                    "llm_component_scores": {
                        "$arrayElemAt": ["$llm_score_data.component_scores", 0]
                    },
                    "llm_recommendation": {
                        "$arrayElemAt": ["$llm_score_data.read_recommendation", 0]
                    },
                }
            },
            # Sort by LLM score (descending) - highest scores first
            {"$sort": {"llm_score": -1}},
            # Take the top articles
            {"$limit": limit * 3},  # Get 3x the requested limit
            # Then take a random subset of the top matches for variety
            {"$sample": {"size": limit}},
            # Project to include only needed fields
            {"$project": {"_id": 0, "llm_score_data": 0}},
        ]

        articles = await db.later.aggregate(pipeline).to_list(length=limit)

        response = {"count": len(articles), "articles": articles}

        return response

    except Exception as e:
        logger.error(f"Error in get_llm_scored_sample: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving LLM scored articles: {str(e)}"
        )


@router.get("/llm-archive", response_model=Dict[str, Any])
async def get_llm_archive_candidates(
    limit: int = Query(
        default=10, ge=1, le=20, description="Number of articles to return"
    ),
    db: AsyncIOMotorDatabase = Depends(get_database),
):
    """
    Get articles to archive based on low LLM scores.
    """
    try:
        # First get articles that have LLM scores
        pipeline = [
            # Join with llm_scores collection
            {
                "$lookup": {
                    "from": "llm_scores",
                    "localField": "id",
                    "foreignField": "article_id",
                    "as": "llm_score_data",
                }
            },
            # Filter to only include articles with LLM scores
            {"$match": {"llm_score_data": {"$ne": []}}},
            # Add LLM score fields to the article
            {
                "$addFields": {
                    "llm_score": {"$arrayElemAt": ["$llm_score_data.llm_score", 0]},
                    "llm_analysis": {"$arrayElemAt": ["$llm_score_data.analysis", 0]},
                    "llm_component_scores": {
                        "$arrayElemAt": ["$llm_score_data.component_scores", 0]
                    },
                    "llm_recommendation": {
                        "$arrayElemAt": ["$llm_score_data.read_recommendation", 0]
                    },
                }
            },
            # Sort by LLM score (ascending) - lowest scores first
            {"$sort": {"llm_score": 1}},
            # Take the bottom articles
            {"$limit": limit * 3},  # Get 3x the requested limit
            # Then take a random subset of the bottom matches for variety
            {"$sample": {"size": limit}},
            # Project to include only needed fields
            {"$project": {"_id": 0, "llm_score_data": 0}},
        ]

        articles = await db.later.aggregate(pipeline).to_list(length=limit)

        response = {"count": len(articles), "articles": articles}

        return response

    except Exception as e:
        logger.error(f"Error in get_llm_archive_candidates: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving LLM archive candidates: {str(e)}"
        )
