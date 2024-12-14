from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PodcastRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = self.db.listened

    async def find_all(self) -> List[Dict[str, Any]]:
        """Fetch all podcasts from the database"""
        try:
            cursor = self.collection.find()
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Error fetching podcasts: {str(e)}")
            raise

    async def count_documents(self) -> int:
        """Count documents in the collection"""
        return await self.collection.count_documents({}) 