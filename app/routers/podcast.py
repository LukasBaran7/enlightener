from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.config import get_podcasts_database
from typing import List, Optional
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel
import dateutil.parser

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/podcasts",
    tags=["podcasts"]
)

class Episode(BaseModel):
    episode_title: str
    audio_url: str
    overcast_url: str
    overcast_id: str
    published_date: str
    play_progress: Optional[int] = None
    last_played_at: Optional[str] = None
    summary: str = ""

class Podcast(BaseModel):
    podcast_title: str
    artwork_url: str
    episodes: List[Episode]
    created_at: str
    source: str = "overcast"

class PodcastService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = self.db.listened

    def process_date(self, date_field: Optional[str | datetime]) -> str:
        """Convert date field to ISO format string"""
        if not date_field:
            return datetime.now(ZoneInfo("UTC")).isoformat()
        
        try:
            if isinstance(date_field, str):
                # Parse the string date using dateutil for better format support
                dt = dateutil.parser.parse(date_field)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                return dt.isoformat()
            elif isinstance(date_field, datetime):
                if date_field.tzinfo is None:
                    date_field = date_field.replace(tzinfo=ZoneInfo("UTC"))
                return date_field.isoformat()
            else:
                logger.warning(f"Unexpected date type: {type(date_field)}")
                return datetime.now(ZoneInfo("UTC")).isoformat()
        except Exception as e:
            logger.error(f"Error processing date {date_field}: {str(e)}")
            return datetime.now(ZoneInfo("UTC")).isoformat()

    async def get_latest_episodes(self) -> List[Podcast]:
        """Fetch latest podcast episodes"""
        try:
            logger.info("Fetching latest episodes")
            
            # Fetch all podcasts from the database
            cursor = self.collection.find()
            podcasts = []
            
            async for doc in cursor:
                logger.info(f"Processing podcast: {doc.get('podcast_title', 'Unknown')}")
                episodes = []
                for episode_doc in doc.get("episodes", []):
                    try:
                        # Log the raw date values for debugging
                        logger.debug(f"Raw published_date: {episode_doc.get('published_date')}")
                        logger.debug(f"Raw last_played_at: {episode_doc.get('last_played_at')}")
                        
                        episode = Episode(
                            episode_title=episode_doc.get("episode_title") or episode_doc.get("title", ""),
                            audio_url=episode_doc.get("audio_url", ""),
                            overcast_url=episode_doc.get("overcast_url", ""),
                            overcast_id=episode_doc.get("overcast_id", ""),
                            published_date=self.process_date(episode_doc.get("published_date")),
                            play_progress=episode_doc.get("play_progress"),
                            last_played_at=self.process_date(episode_doc.get("last_played_at")),
                            summary=episode_doc.get("summary", "")
                        )
                        episodes.append(episode)
                    except Exception as e:
                        logger.error(f"Error processing episode: {str(e)}")
                        continue

                try:
                    podcast = Podcast(
                        podcast_title=doc.get("podcast_title", ""),
                        artwork_url=doc.get("artwork_url", ""),
                        episodes=episodes,
                        created_at=self.process_date(doc.get("created_at")),
                        source=doc.get("source", "overcast")
                    )
                    podcasts.append(podcast)
                except Exception as e:
                    logger.error(f"Error processing podcast: {str(e)}")
                    continue

            # Get latest episodes across all podcasts
            all_episodes = []
            for podcast in podcasts:
                for episode in podcast.episodes:
                    all_episodes.append((episode, podcast))

            # Sort episodes by last_played_at and take the 15 most recent
            all_episodes.sort(
                key=lambda x: dateutil.parser.parse(x[0].last_played_at or "1970-01-01T00:00:00+00:00"),
                reverse=True
            )
            latest_episodes = all_episodes[:15]

            # Group episodes by podcast
            podcast_map = {}
            for episode, podcast in latest_episodes:
                if podcast.podcast_title not in podcast_map:
                    podcast_map[podcast.podcast_title] = podcast.copy(
                        update={"episodes": []}
                    )
                podcast_map[podcast.podcast_title].episodes.append(episode)

            result = list(podcast_map.values())
            return result

        except Exception as e:
            logger.error(f"Error in get_latest_episodes: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching podcasts: {str(e)}"
            )

@router.get("/latest", response_model=List[Podcast])
async def get_latest_episodes(
    db: AsyncIOMotorDatabase = Depends(get_podcasts_database)
):
    """Get the 15 most recently played episodes grouped by podcast"""
    try:
        podcast_service = PodcastService(db)
        return await podcast_service.get_latest_episodes()
    except Exception as e:
        logger.error(f"Error in get_latest_episodes endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching podcasts: {str(e)}"
        )

@router.get("/check")
async def check_podcast_collection(
    db: AsyncIOMotorDatabase = Depends(get_podcasts_database)
):
    """Check if the podcast collection is accessible and contains data"""
    try:
        count = await db.listened.count_documents({})
        return {
            "collection_exists": True,
            "document_count": count
        }
    except Exception as e:
        return {
            "collection_exists": False,
            "error": str(e)
        } 