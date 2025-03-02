from fastapi import APIRouter, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase
from app.core.config import get_podcasts_database
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from pydantic import BaseModel


from app.repositories.podcast_repository import PodcastRepository
import dateutil.parser


# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/podcasts", tags=["podcasts"])


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
    def __init__(self, repository: PodcastRepository):
        self.repository = repository

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

    def _create_episode(self, episode_doc: Dict[str, Any]) -> Episode:
        """Create Episode model from document"""
        return Episode(
            episode_title=episode_doc.get("episode_title")
            or episode_doc.get("title", ""),
            audio_url=episode_doc.get("audio_url", ""),
            overcast_url=episode_doc.get("overcast_url", ""),
            overcast_id=episode_doc.get("overcast_id", ""),
            published_date=self.process_date(episode_doc.get("published_date")),
            play_progress=episode_doc.get("play_progress"),
            last_played_at=self.process_date(episode_doc.get("last_played_at")),
            summary=episode_doc.get("summary", ""),
        )

    def _create_podcast(self, doc: Dict[str, Any], episodes: List[Episode]) -> Podcast:
        """Create Podcast model from document"""
        return Podcast(
            podcast_title=doc.get("podcast_title", ""),
            artwork_url=doc.get("artwork_url", ""),
            episodes=episodes,
            created_at=self.process_date(doc.get("created_at")),
            source=doc.get("source", "overcast"),
        )

    async def get_latest_episodes(self) -> List[Podcast]:
        """Fetch latest podcast episodes"""
        try:
            podcasts_data = await self.repository.find_all()
            podcasts = []

            for doc in podcasts_data:
                episodes = []
                for episode_doc in doc.get("episodes", []):
                    try:
                        episode = self._create_episode(episode_doc)
                        episodes.append(episode)
                    except Exception as e:
                        logger.error(f"Error processing episode: {str(e)}")
                        continue

                try:
                    podcast = self._create_podcast(doc, episodes)
                    podcasts.append(podcast)
                except Exception as e:
                    logger.error(f"Error processing podcast: {str(e)}")
                    continue

            # Get latest episodes across all podcasts
            all_episodes = [
                (episode, podcast)
                for podcast in podcasts
                for episode in podcast.episodes
            ]

            # Sort episodes by last_played_at
            def parse_date(date_str: Optional[str]) -> datetime:
                if not date_str:
                    return datetime.fromtimestamp(0, UTC)
                try:
                    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except ValueError:
                    return datetime.fromtimestamp(0, UTC)

            all_episodes.sort(
                key=lambda x: parse_date(x[0].last_played_at),
                reverse=True,
            )
            # Return all episodes instead of limiting to 50
            latest_episodes = all_episodes

            # Group episodes by podcast
            podcast_map = {}
            for episode, podcast in latest_episodes:
                if podcast.podcast_title not in podcast_map:
                    podcast_map[podcast.podcast_title] = podcast.model_copy(
                        update={"episodes": []}
                    )
                podcast_map[podcast.podcast_title].episodes.append(episode)

            return list(podcast_map.values())

        except Exception as e:
            logger.error(f"Error in get_latest_episodes: {str(e)}")
            raise


# Update the endpoint dependencies
def get_podcast_service(
    db: AsyncIOMotorDatabase = Depends(get_podcasts_database),
) -> PodcastService:
    repository = PodcastRepository(db)
    return PodcastService(repository)


@router.get("/latest", response_model=List[Podcast])
async def get_latest_episodes(service: PodcastService = Depends(get_podcast_service)):
    """Get all played episodes grouped by podcast, sorted by most recently played"""
    try:
        return await service.get_latest_episodes()
    except Exception as e:
        logger.error(f"Error in get_latest_episodes endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching podcasts: {str(e)}"
        )


@router.get("/check")
async def check_podcast_collection(
    repository: PodcastRepository = Depends(
        lambda db=Depends(get_podcasts_database): PodcastRepository(db)
    ),
):
    """Check if the podcast collection is accessible and contains data"""
    try:
        count = await repository.count_documents()
        return {"collection_exists": True, "document_count": count}
    except Exception as e:
        return {"collection_exists": False, "error": str(e)}
