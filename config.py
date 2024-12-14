import os
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:5173")


def get_cors_origins() -> List[str]:
    return CORS_ORIGINS_STR.split(",")


IS_DEVELOPMENT = ENVIRONMENT == "development"
