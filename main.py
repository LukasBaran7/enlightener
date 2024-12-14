from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import get_cors_origins

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
