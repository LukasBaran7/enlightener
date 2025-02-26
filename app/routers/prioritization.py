from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database
from app.services.content_extractor import ContentExtractor
from bson import ObjectId
import logging

router = APIRouter(prefix="/prioritization", tags=["prioritization"])
logger = logging.getLogger(__name__)


class PrioritizationService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.content_extractor = ContentExtractor()

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

        # Convert ObjectId to string for JSON serialization
        for article in processed_articles:
            if "_id" in article:
                article["_id"] = str(article["_id"])

        return processed_articles

    except Exception as e:
        logger.error(f"Error in get_prioritization_sample: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving prioritization sample: {str(e)}"
        )
