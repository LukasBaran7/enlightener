from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database
from bson import ObjectId
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

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

    async def get_random_articles(self, collection_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch random articles from specified collection, with at least half from before 2020"""
        # First, get IDs of recently shown articles (from last 2 hours)
        two_hours_ago = datetime.now(ZoneInfo("UTC")) - timedelta(hours=2)
        recently_shown = await self.db.to_read.find({
            "shown_at": {
                "$gte": two_hours_ago.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }
        }).distinct("article_id")

        # Calculate how many old articles we need
        old_articles_limit = limit // 2

        # Pipeline for old articles (before 2020)
        old_articles_pipeline = [
            {
                "$match": {
                    "_id": {"$nin": recently_shown},
                    "saved_at": {
                        "$lt": "2020-01-01T00:00:00.000+00:00"
                    }
                }
            },
            {"$sample": {"size": old_articles_limit}}
        ]

        # Pipeline for newer articles
        new_articles_pipeline = [
            {
                "$match": {
                    "_id": {"$nin": recently_shown},
                    "saved_at": {
                        "$gte": "2020-01-01T00:00:00.000+00:00"
                    }
                }
            },
            {"$sample": {"size": limit - old_articles_limit}}
        ]

        # Get old articles
        old_cursor = self.db[collection_name].aggregate(old_articles_pipeline)
        old_articles = await old_cursor.to_list(length=None)

        # Get newer articles
        new_cursor = self.db[collection_name].aggregate(new_articles_pipeline)
        new_articles = await new_cursor.to_list(length=None)

        # Combine articles
        articles = old_articles + new_articles

        # If we didn't get enough articles, fill with random ones without date restriction
        if len(articles) < limit:
            remaining = limit - len(articles)
            existing_ids = {article["_id"] for article in articles}
            fallback_pipeline = [
                {
                    "$match": {
                        "_id": {
                            "$nin": list(existing_ids) + recently_shown
                        }
                    }
                },
                {"$sample": {"size": remaining}}
            ]
            additional_cursor = self.db[collection_name].aggregate(fallback_pipeline)
            additional_articles = await additional_cursor.to_list(length=None)
            articles.extend(additional_articles)

        # Record these articles as recently shown
        now = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
        await self.db.to_read.insert_many([
            {
                "article_id": article["_id"],
                "shown_at": now
            } for article in articles
        ])

        # Clean up old entries
        await self.db.to_read.delete_many({
            "shown_at": {
                "$lt": two_hours_ago.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }
        })

        return articles

    async def get_daily_article_counts(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get count of articles read per day for the last N days"""
        # Calculate date threshold
        end_date = datetime.now(ZoneInfo("UTC"))
        start_date = end_date - timedelta(days=days)
        
        # Create aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "updated_at": {
                        "$gte": start_date.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
                        "$lte": end_date.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    },
                    # Exclude newsletters by checking category
                    "category": {"$ne": "email"}
                }
            },
            {
                "$group": {
                    "_id": {
                        "$substr": ["$updated_at", 0, 10]  # Group by date (YYYY-MM-DD)
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}  # Sort by date ascending
            }
        ]
        
        # Execute aggregation
        cursor = self.db.archived.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        # Fill in missing dates with zero counts
        date_counts = defaultdict(int)
        for result in results:
            date_counts[result["_id"]] = result["count"]
        
        # Generate all dates in range
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            all_dates.append({
                "date": date_str,
                "count": date_counts[date_str]
            })
            current_date += timedelta(days=1)
        
        return all_dates

    async def get_later_article_counts(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get count of articles added to 'later' collection per day for the last N days"""
        # Calculate date threshold
        end_date = datetime.now(ZoneInfo("UTC"))
        start_date = end_date - timedelta(days=days)
        
        # Create aggregation pipeline
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": start_date.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),
                        "$lte": end_date.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "$substr": ["$created_at", 0, 10]  # Group by date (YYYY-MM-DD)
                    },
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"_id": 1}  # Sort by date ascending
            }
        ]
        
        # Execute aggregation
        cursor = self.db.later.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        # Fill in missing dates with zero counts
        date_counts = defaultdict(int)
        for result in results:
            date_counts[result["_id"]] = result["count"]
        
        # Generate all dates in range
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            all_dates.append({
                "date": date_str,
                "count": date_counts[date_str]
            })
            current_date += timedelta(days=1)
        
        return all_dates

    async def get_collection_counts(self) -> Dict[str, int]:
        """Get total counts for archived and later collections"""
        archived_count = await self.db.archived.count_documents({})
        later_count = await self.db.later.count_documents({})
        return {
            "archived_count": archived_count,
            "later_count": later_count
        }

    async def get_curated_articles(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get curated articles in different sections"""
        two_hours_ago = datetime.now(ZoneInfo("UTC")) - timedelta(hours=2)
        recently_shown = await self.db.to_read.find({
            "shown_at": {
                "$gte": two_hours_ago.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }
        }).distinct("article_id")

        # Get quick reads
        quick_reads_pipeline = [
            {
                "$match": {
                    "_id": {"$nin": recently_shown},
                    "word_count": {"$lt": 1000},
                    "word_count": {"$gt": 0}
                }
            },
            {"$sample": {"size": 4}}
        ]

        # Get old articles (2020 and older)
        archive_pipeline = [
            {
                "$match": {
                    "_id": {"$nin": recently_shown},
                    "saved_at": {
                        "$lt": "2021-01-01T00:00:00.000+00:00"  # Articles before 2021
                    }
                }
            },
            {
                "$sort": {"saved_at": -1}  # Sort by saved_at descending
            },
            {"$sample": {"size": 4}}  # Get 4 random old articles
        ]

        # Get most common sites
        common_sites_pipeline = [
            {
                "$group": {
                    "_id": "$site_name",
                    "count": {"$sum": 1}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        
        common_sites = await self.db.later.aggregate(common_sites_pipeline).to_list(length=None)
        common_site_names = [site["_id"] for site in common_sites if site["_id"]]

        # Get articles from favorite sources
        favorite_sources_pipeline = [
            {
                "$match": {
                    "_id": {"$nin": recently_shown},
                    "site_name": {"$in": common_site_names}
                }
            },
            {"$sample": {"size": 3}}
        ]

        # Execute all pipelines
        quick_reads = await self.db.later.aggregate(quick_reads_pipeline).to_list(length=None)
        archive_articles = await self.db.later.aggregate(archive_pipeline).to_list(length=None)
        favorite_sources = await self.db.later.aggregate(favorite_sources_pipeline).to_list(length=None)

        # Record all returned articles as recently shown
        all_articles = quick_reads + archive_articles + favorite_sources
        if all_articles:
            now = datetime.now(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            await self.db.to_read.insert_many([
                {
                    "article_id": article["_id"],
                    "shown_at": now
                } for article in all_articles
            ])

        # Clean up old entries
        await self.db.to_read.delete_many({
            "shown_at": {
                "$lt": two_hours_ago.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }
        })

        return {
            "quick_reads": [process_mongodb_doc(doc) for doc in quick_reads],
            "from_archives": [process_mongodb_doc(doc) for doc in archive_articles],
            "favorite_sources": [process_mongodb_doc(doc) for doc in favorite_sources]
        }

    async def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get combined statistics including collection counts and daily counts"""
        # Get collection counts
        archived_count = await self.db.archived.count_documents({})
        later_count = await self.db.later.count_documents({})
        
        # Get daily counts for archived articles
        archived_daily = await self.get_daily_article_counts(days)
        
        # Get daily counts for later articles
        later_daily = await self.get_later_article_counts(days)
        
        return {
            "total_counts": {
                "archived_count": archived_count,
                "later_count": later_count
            },
            "daily_counts": {
                "archived": archived_daily,
                "later": later_daily
            }
        }

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
    limit: int = Query(default=500, ge=1, le=500, description="Number of articles to return"),
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

@router.get("/later/curated")
async def get_curated_later_articles(
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get curated articles from the 'later' collection in different sections"""
    try:
        article_service = ArticleService(db)
        curated_articles = await article_service.get_curated_articles()
        return curated_articles
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving curated articles: {str(e)}"
        )

@router.get("/stats")
async def get_stats(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to look back"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get combined statistics including collection counts and daily counts"""
    try:
        article_service = ArticleService(db)
        stats = await article_service.get_stats(days)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )