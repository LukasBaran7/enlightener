from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import reader
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Reader API",
    description="API for managing articles and reading list",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reader.router)

@app.get("/")
async def root():
    return {"message": "Welcome to Reader API"} 