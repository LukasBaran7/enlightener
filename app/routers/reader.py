from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.models.article import Article
from app.core.config import get_database
from bson import ObjectId
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # For proper timezone handling

router = APIRouter(
    prefix="/reader",
    tags=["reader"]
)

def convert_objectid(obj):
    """Convert ObjectId to string in MongoDB document"""
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

def process_mongodb_doc(doc):
    """Process MongoDB document to handle ObjectId and other special types"""
    if doc.get('_id'):
        doc['_id'] = str(doc['_id'])
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
    """
    Retrieve articles from the archived collection with pagination and sorting
    """
    try:
        # Build the query
        query = {}
        if search:
            query["$or"] = [
                {"title": {"$regex": search, "$options": "i"}},
                {"summary": {"$regex": search, "$options": "i"}}
            ]
        
        # Add date filter if days parameter is provided
        if days:
            # Use UTC for consistency with MongoDB
            date_threshold = datetime.now(ZoneInfo("UTC")) - timedelta(days=days)
            # Format date to match MongoDB string format: "2024-12-01T22:35:07.522248+00:00"
            query["updated_at"] = {
                "$gte": date_threshold.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            }

        # Log the query for debugging
        print(f"MongoDB Query: {query}")
        
        # Sort direction
        sort_direction = -1 if order == "desc" else 1
        
        # Get total count for pagination
        total_count = await db.archived.count_documents(query)
        print(f"Total matching documents: {total_count}")
        
        # Execute query with pagination and sorting
        cursor = db.archived.find(query)
        cursor = cursor.sort(sort_by, sort_direction).skip(skip).limit(limit)
        
        # Convert to list and process documents
        articles = await cursor.to_list(length=None)
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
async def get_article(article_id: str, db: AsyncIOMotorDatabase = Depends(get_database)):
    """
    Retrieve a specific article by ID from archived collection
    """
    try:
        if not ObjectId.is_valid(article_id):
            raise HTTPException(status_code=400, detail="Invalid article ID format")

        article = await db.archived.find_one({"_id": ObjectId(article_id)})
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return process_mongodb_doc(article)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving article: {str(e)}"
        )