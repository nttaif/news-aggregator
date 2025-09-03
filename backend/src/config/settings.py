from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    #App settings
    app_name: str = "News Aggregator"
    version: str = "1.0.0"
    debug: bool = False

    #Database settings
    database_url: str
    redis_url: str = None

    #API settings
    news_api_key: str
    guardian_api_key: Optional[str] = None
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    google_translate_api_key: Optional[str] = None

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Cache
    cache_expire_time: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
    
settings = Settings()
    


