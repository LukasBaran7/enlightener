from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database
from bson import ObjectId
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

router = APIRouter(
    prefix="/reader",
    tags=["reader"]
)

class ArticleQueryBuilder:
    def __init__(self):
        self.query: Dict[str, Any] = {}

    def add_search_filter(self, search: Optional[str]) -> 'ArticleQueryBuilder':
        if search:
            self.query["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"summary": {"$regex": search, "$options": "i"}}
            ]
        return self

    def add_date_filter(self, days: Optional[int]) -> 'ArticleQueryBuilder':
        if days:
            date_threshold = datetime.now(ZoneInfo("UTC")) - timedelta(days=days)
            self.query["updated_at"] = {
                "$gte": date_threshold.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }
        return self

    def build(self) -> Dict[str, Any]:
        return self.query

class ArticleService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db

    async def get_articles_count(self, query: Dict[str, Any]) -> int:
        return await self.db.archived.count_documents(query)

    def get_sort_direction(self, order: str) -> int:
        return -1 if order == "desc" else 1

    async def fetch_articles(
        self,
        query: Dict[str, Any],
        sort_by: str,
        sort_direction: int,
        skip: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        cursor = self.db.archived.find(query).sort(sort_by, sort_direction)
        
        if skip is not None and limit is not None:
            cursor = cursor.skip(skip).limit(limit)
            
        return await cursor.to_list(length=None)

    async def get_article_by_id(self, article_id: str) -> Dict[str, Any]:
        if not ObjectId.is_valid(article_id):
            raise HTTPException(status_code=400, detail="Invalid article ID format")

        article = await self.db.archived.find_one({"_id": ObjectId(article_id)})
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return process_mongodb_doc(article)

    async def get_random_articles(self, collection_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch random articles from specified collection"""
        # Using MongoDB's aggregation pipeline with $sample operator
        pipeline = [
            {"$sample": {"size": limit}}
        ]
        cursor = self.db[collection_name].aggregate(pipeline)
        return await cursor.to_list(length=None)

def process_mongodb_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Process MongoDB document to handle ObjectId and other special types"""
    # Convert ObjectId to string
    if doc.get('_id'):
        doc['_id'] = str(doc['_id'])
    
    # Set default values for required fields if they're missing or None
    if 'word_count' not in doc or doc['word_count'] is None:
        doc['word_count'] = 0
    
    # Ensure other required fields have default values
    if 'title' not in doc:
        doc['title'] = ''
    if 'url' not in doc:
        doc['url'] = ''
    if 'summary' not in doc:
        doc['summary'] = ''
    if 'site_name' not in doc or doc['site_name'] is None:
        doc['site_name'] = ''
    if 'created_at' not in doc:
        doc['created_at'] = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    if 'updated_at' not in doc:
        doc['updated_at'] = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    
    return doc

@router.get("/articles", response_model=List[Article])
async def get_articles(
    skip: int = Query(default=0, ge=0, description="Number of articles to skip"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of articles to return"),
    sort_by: Optional[str] = Query(default="updated_at", description="Field to sort by"),
    order: Optional[str] = Query(default="desc", enum=["asc", "desc"]),
    search: Optional[str] = Query(default=None, description="Search in title and summary"),
    days: Optional[int] = Query(default=None, description="Get articles from last N days"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve articles from the archived collection with pagination and sorting"""
    try:
        article_service = ArticleService(db)
        
        # Build query using builder pattern
        query = (ArticleQueryBuilder()
                .add_search_filter(search)
                .add_date_filter(days)
                .build())

        # Get total count and sort direction
        total_count = await article_service.get_articles_count(query)
        sort_direction = article_service.get_sort_direction(order)

        # Determine if we should use pagination
        use_pagination = not (days and days <= 14 and total_count <= 100)
        
        # Fetch articles
        articles = await article_service.fetch_articles(
            query=query,
            sort_by=sort_by,
            sort_direction=sort_direction,
            skip=skip if use_pagination else None,
            limit=limit if use_pagination else None
        )
        
        processed_articles = [process_mongodb_doc(doc) for doc in articles]
        print(f"Returned articles count: {len(processed_articles)}")
        
        return processed_articles

    except Exception as e:
        print(f"Error in get_articles: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving articles: {str(e)}"
        )

@router.get("/articles/{article_id}", response_model=Article)
async def get_article(
    article_id: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve a specific article by ID from archived collection"""
    try:
        article_service = ArticleService(db)
        return await article_service.get_article_by_id(article_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving article: {str(e)}"
        )

@router.get("/random-later", response_model=List[Article])
async def get_random_later_articles(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Retrieve 5 random articles from the 'later' collection"""
    try:
        article_service = ArticleService(db)
        articles = await article_service.get_random_articles("later")
        processed_articles = [process_mongodb_doc(doc) for doc in articles]
        return processed_articles
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving random articles: {str(e)}"
        )