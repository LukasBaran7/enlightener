from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Depends, HTTPException
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
from typing import List
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    MONGODB_URL: str = os.getenv("MONGODB_URL", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "")
    PODCASTS_DATABASE_NAME: str = os.getenv("PODCASTS_DATABASE_NAME", "podcasts")
    ENVIRONMENT: str
    CORS_ORIGINS: str
    READWISE_TOKEN: str
    ANTHROPIC_API_KEY: str

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    def validate(self):
        if not self.MONGODB_URL:
            raise ValueError("MONGODB_URL is not set")
        if not self.DATABASE_NAME:
            raise ValueError("DATABASE_NAME is not set")


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings


async def get_database(settings: Settings = Depends(get_settings)):
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        # Verify the connection
        await client.admin.command("ping")
        db = client[settings.DATABASE_NAME]

        # Create TTL index on to_read collection
        await db.to_read.create_index("shown_at", expireAfterSeconds=7200)  # 2 hours

        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not connect to database")


async def get_podcasts_database(settings: Settings = Depends(get_settings)):
    try:
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        # Verify the connection
        await client.admin.command("ping")
        return client[settings.PODCASTS_DATABASE_NAME]
    except Exception as e:
        logger.error(f"Failed to connect to Podcasts MongoDB: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Could not connect to podcasts database"
        )
