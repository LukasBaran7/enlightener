from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class Article(BaseModel):
    """
    Article model that matches MongoDB document structure.
    """

    model_config = ConfigDict(
        populate_by_name=True, json_encoders={datetime: lambda v: v.isoformat()}
    )

    id: str = Field(default=None, alias="_id")
    readwise_id: str = Field(..., alias="id")
    url: str
    title: str
    author: str
    source: Optional[str] = None
    category: str
    location: str
    tags: Dict[str, Any] = Field(default_factory=dict)
    site_name: str
    word_count: int = Field(..., alias="word_count")
    created_at: datetime
    updated_at: datetime
    published_date: Optional[int] = Field(
        None, description="Unix timestamp in milliseconds"
    )
    summary: Optional[str] = None
    image_url: Optional[str] = None
    content: Optional[str] = None
    source_url: Optional[str] = None
    notes: str = ""
    parent_id: Optional[str] = None
    reading_progress: float
    first_opened_at: Optional[datetime] = None
    last_opened_at: Optional[datetime] = None
    saved_at: datetime
    last_moved_at: datetime
