# Backend Implementation Guide - News Aggregator & Sentiment Analysis

## Mục Lục
1. [Cài Đặt Môi Trường](#1-cài-đặt-môi-trường)
2. [Cấu Trúc Dự Án](#2-cấu-trúc-dự-án)
3. [Cài Đặt Dependencies](#3-cài-đặt-dependencies)
4. [Cấu Hình Database](#4-cấu-hình-database)
5. [Xây Dựng API Endpoints](#5-xây-dựng-api-endpoints)
6. [News Collection System](#6-news-collection-system)
7. [Sentiment Analysis Engine](#7-sentiment-analysis-engine)
8. [Search & Recommendation](#8-search--recommendation)
9. [Caching System](#9-caching-system)
10. [Authentication & Security](#10-authentication--security)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)

---

## 1. Cài Đặt Môi Trường

### Bước 1.1: Kiểm tra Python
```bash
python --version  # Cần Python 3.8+
```

### Bước 1.2: Tạo Virtual Environment
```bash
cd news-aggregator/backend
python -m venv .venv

# Kích hoạt virtual environment
# Windows:
.venv\Scripts\activate

### Bước 1.3: Tạo file .env
```bash
# Tạo file .env trong thư mục backend/
touch .env  # Linux/Mac
# hoặc tạo file .env bằng editor
```

**Nội dung file .env:**
```env
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/news_aggregator
REDIS_URL=redis://localhost:6379

# News APIs
NEWS_API_KEY=your_newsapi_key_here
GUARDIAN_API_KEY=your_guardian_key_here
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Translation
GOOGLE_TRANSLATE_API_KEY=your_google_translate_key

# Security
SECRET_KEY=your_super_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Environment
ENVIRONMENT=development
DEBUG=True
```

---

## 2. Cấu Trúc Dự Án

### Bước 2.1: Tạo cấu trúc thư mục
```
backend/
├── .env
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── article.py
│   │   ├── user.py
│   │   └── sentiment.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── article.py
│   │   ├── user.py
│   │   └── search.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── news.py
│   │       ├── search.py
│   │       ├── sentiment.py
│   │       └── auth.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── news_collector.py
│   │   ├── sentiment_analyzer.py
│   │   ├── search_engine.py
│   │   └── cache_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── security.py
│   │   ├── helpers.py
│   │   └── validators.py
│   └── tasks/
│       ├── __init__.py
│       ├── scheduler.py
│       └── background_jobs.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_services.py
│   └── test_models.py
├── migrations/
├── scripts/
│   ├── init_db.py
│   └── populate_data.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

### Bước 2.2: Tạo các thư mục
```bash
mkdir -p src/{config,models,schemas,api/v1,services,utils,tasks}
mkdir -p tests migrations scripts docker
touch src/__init__.py src/config/__init__.py src/models/__init__.py
touch src/schemas/__init__.py src/api/__init__.py src/api/v1/__init__.py
touch src/services/__init__.py src/utils/__init__.py src/tasks/__init__.py
touch tests/__init__.py
```

---

## 3. Cài Đặt Dependencies

### Bước 3.1: Cập nhật requirements.txt
```txt
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.7
alembic==1.12.1

# Redis
redis==5.0.1
python-redis-lock==4.0.0

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Environment
python-dotenv==1.0.0
pydantic-settings==2.0.3

# HTTP Clients
httpx==0.25.2
aiohttp==3.9.1

# News APIs
newsapi-python==0.2.6
python-reddit-api-wrapper==0.4.0

# ML & NLP
transformers==4.35.2
torch==2.1.1
sentence-transformers==2.2.2
spacy==3.7.2
scikit-learn==1.3.2

# Vietnamese NLP
py_vncorenlp==1.0.3

# Translation
google-cloud-translate==3.12.1

# Search
elasticsearch==8.11.0
faiss-cpu==1.7.4

# Background Tasks
celery==5.3.4
redis==5.0.1

# Monitoring
prometheus-client==0.19.0

# Utilities
python-dateutil==2.8.2
pytz==2023.3
requests==2.31.0
beautifulsoup4==4.12.2
feedparser==6.0.10

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
mypy==1.7.1

# Logging
structlog==23.2.0
```

### Bước 3.2: Cài đặt dependencies
```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

---

## 4. Cấu Hình Database

### Bước 4.1: Tạo file config/settings.py
```python
from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # App
    app_name: str = "News Aggregator API"
    version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str
    redis_url: str = "redis://localhost:6379"
    
    # APIs
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
```

### Bước 4.2: Tạo file config/database.py
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings

# Database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for getting DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Bước 4.3: Tạo Models

**models/article.py:**
```python
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
from ..config.database import Base

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text)
    summary = Column(Text)
    url = Column(String(1000), unique=True, nullable=False)
    source = Column(String(100), nullable=False, index=True)
    author = Column(String(100))
    
    # Timestamps
    published_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Sentiment Analysis
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive/negative/neutral
    emotional_intensity = Column(Float)
    sentiment_confidence = Column(Float)
    
    # Categorization
    category = Column(String(50), index=True)
    tags = Column(ARRAY(String))
    language = Column(String(10), default='en')
    
    # Engagement metrics
    view_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    trending_score = Column(Float, default=0, index=True)
    
    # Additional metadata
    metadata = Column(JSONB)
    is_active = Column(Boolean, default=True)
    
    # Search
    search_vector = Column(Text)  # For full-text search
    
    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title[:50]}...')>"
```

**models/user.py:**
```python
from sqlalchemy import Column, String, Boolean, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid
from ..config.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Preferences
    preferred_categories = Column(ARRAY(String))
    preferred_sources = Column(ARRAY(String))
    preferred_languages = Column(ARRAY(String), default=['en'])
    sentiment_preference = Column(String(20), default='all')
    
    # Activity tracking
    interaction_history = Column(JSONB)
    last_login = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"
```

---

## 5. Xây Dựng API Endpoints

### Bước 5.1: Tạo Schemas

**schemas/article.py:**
```python
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

class ArticleBase(BaseModel):
    title: str = Field(..., max_length=500)
    content: Optional[str] = None
    summary: Optional[str] = None
    url: HttpUrl
    source: str = Field(..., max_length=100)
    author: Optional[str] = Field(None, max_length=100)
    published_at: Optional[datetime] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = []
    language: str = Field('en', max_length=10)

class ArticleCreate(ArticleBase):
    pass

class ArticleUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    content: Optional[str] = None
    summary: Optional[str] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None

class SentimentData(BaseModel):
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    sentiment_label: Optional[str] = Field(None, regex="^(positive|negative|neutral)$")
    emotional_intensity: Optional[float] = Field(None, ge=0, le=1)
    sentiment_confidence: Optional[float] = Field(None, ge=0, le=1)

class ArticleResponse(ArticleBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    emotional_intensity: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    view_count: int = 0
    share_count: int = 0
    trending_score: float = 0
    is_active: bool = True
    
    class Config:
        from_attributes = True

class ArticleListResponse(BaseModel):
    articles: List[ArticleResponse]
    total: int
    page: int
    per_page: int
    pages: int

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    category: Optional[str] = None
    source: Optional[str] = None
    sentiment: Optional[str] = Field(None, regex="^(positive|negative|neutral)$")
    language: Optional[str] = 'en'
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    semantic: bool = False
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
```

### Bước 5.2: Tạo API Routes

**api/v1/news.py:**
```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from ...config.database import get_db
from ...schemas.article import ArticleResponse, ArticleListResponse, SearchRequest
from ...services.news_collector import NewsCollectorService
from ...services.search_engine import SearchEngine
from ...models.article import Article
import math

router = APIRouter(prefix="/news", tags=["news"])

@router.get("/", response_model=ArticleListResponse)
async def get_articles(
    category: Optional[str] = Query(None, description="Filter by category"),
    source: Optional[str] = Query(None, description="Filter by source"),
    sentiment: Optional[str] = Query(None, regex="^(positive|negative|neutral)$"),
    language: Optional[str] = Query('en', description="Language filter"),
    limit: int = Query(20, ge=1, le=100, description="Number of articles"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Lấy danh sách bài viết với filtering và pagination
    """
    query = db.query(Article).filter(Article.is_active == True)
    
    # Apply filters
    if category:
        query = query.filter(Article.category == category)
    if source:
        query = query.filter(Article.source == source)
    if sentiment:
        query = query.filter(Article.sentiment_label == sentiment)
    if language:
        query = query.filter(Article.language == language)
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering
    articles = query.order_by(Article.published_at.desc())\
                   .offset(offset)\
                   .limit(limit)\
                   .all()
    
    return ArticleListResponse(
        articles=articles,
        total=total,
        page=math.ceil(offset / limit) + 1,
        per_page=limit,
        pages=math.ceil(total / limit)
    )

@router.get("/trending", response_model=ArticleListResponse)
async def get_trending_articles(
    timeframe: str = Query('24h', regex="^(1h|6h|24h|7d)$"),
    category: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Lấy bài viết trending theo thời gian
    """
    from datetime import datetime, timedelta
    
    # Calculate time threshold
    time_map = {'1h': 1, '6h': 6, '24h': 24, '7d': 168}
    hours_ago = datetime.utcnow() - timedelta(hours=time_map[timeframe])
    
    query = db.query(Article)\
              .filter(Article.is_active == True)\
              .filter(Article.published_at >= hours_ago)
    
    if category:
        query = query.filter(Article.category == category)
    
    articles = query.order_by(Article.trending_score.desc())\
                   .limit(limit)\
                   .all()
    
    return ArticleListResponse(
        articles=articles,
        total=len(articles),
        page=1,
        per_page=limit,
        pages=1
    )

@router.get("/search", response_model=ArticleListResponse)
async def search_articles(
    q: str = Query(..., min_length=1, description="Search query"),
    semantic: bool = Query(False, description="Enable semantic search"),
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    search_engine: SearchEngine = Depends()
):
    """
    Tìm kiếm bài viết với full-text search hoặc semantic search
    """
    if semantic:
        # Use semantic search
        results = await search_engine.semantic_search(
            query=q,
            filters={
                'category': category,
                'sentiment': sentiment,
                'date_from': date_from,
                'date_to': date_to
            },
            limit=limit,
            offset=offset
        )
        return results
    else:
        # Use traditional search
        query = db.query(Article).filter(Article.is_active == True)
        
        # Full-text search in title and content
        search_filter = Article.title.ilike(f"%{q}%") | \
                       Article.content.ilike(f"%{q}%")
        query = query.filter(search_filter)
        
        # Apply additional filters
        if category:
            query = query.filter(Article.category == category)
        if sentiment:
            query = query.filter(Article.sentiment_label == sentiment)
        
        total = query.count()
        articles = query.order_by(Article.published_at.desc())\
                       .offset(offset)\
                       .limit(limit)\
                       .all()
        
        return ArticleListResponse(
            articles=articles,
            total=total,
            page=math.ceil(offset / limit) + 1,
            per_page=limit,
            pages=math.ceil(total / limit)
        )

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: str,
    db: Session = Depends(get_db)
):
    """
    Lấy chi tiết một bài viết
    """
    article = db.query(Article).filter(Article.id == article_id).first()
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    # Increment view count
    article.view_count += 1
    db.commit()
    
    return article

@router.post("/collect")
async def trigger_news_collection(
    sources: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    collector: NewsCollectorService = Depends()
):
    """
    Trigger manual news collection (admin only)
    """
    await collector.collect_news(sources=sources, categories=categories)
    return {"message": "News collection triggered successfully"}
```

---

## 6. News Collection System

### Bước 6.1: Tạo file services/news_collector.py
```python
import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from sqlalchemy.orm import Session
from ..config.settings import settings
from ..config.database import SessionLocal
from ..models.article import Article
from ..services.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class NewsCollectorService:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=settings.news_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    async def collect_news(
        self, 
        sources: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        max_articles: int = 100
    ):
        """
        Thu thập tin tức từ nhiều nguồn
        """
        logger.info("Starting news collection...")
        
        # Default categories if not specified
        if not categories:
            categories = ['technology', 'business', 'health', 'science', 'sports']
        
        collected_articles = []
        
        # Collect from NewsAPI
        if not sources or 'newsapi' in sources:
            articles = await self._collect_from_newsapi(categories, max_articles // 2)
            collected_articles.extend(articles)
        
        # Collect from Guardian (if API key available)
        if settings.guardian_api_key and (not sources or 'guardian' in sources):
            articles = await self._collect_from_guardian(categories, max_articles // 4)
            collected_articles.extend(articles)
        
        # Collect from RSS feeds
        if not sources or 'rss' in sources:
            articles = await self._collect_from_rss(max_articles // 4)
            collected_articles.extend(articles)
        
        # Process and save articles
        saved_count = await self._process_and_save_articles(collected_articles)
        
        logger.info(f"News collection completed. Saved {saved_count} new articles.")
        return saved_count
    
    async def _collect_from_newsapi(
        self, 
        categories: List[str], 
        max_articles: int
    ) -> List[Dict[str, Any]]:
        """
        Thu thập từ NewsAPI
        """
        articles = []
        articles_per_category = max_articles // len(categories)
        
        for category in categories:
            try:
                response = self.newsapi.get_top_headlines(
                    category=category,
                    language='en',
                    page_size=min(articles_per_category, 100)
                )
                
                for article_data in response.get('articles', []):
                    if self._is_valid_article(article_data):
                        articles.append({
                            'title': article_data['title'],
                            'content': article_data.get('content', ''),
                            'url': article_data['url'],
                            'source': article_data['source']['name'],
                            'author': article_data.get('author'),
                            'published_at': self._parse_datetime(article_data['publishedAt']),
                            'category': category,
                            'language': 'en'
                        })
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error collecting from NewsAPI category {category}: {e}")
        
        return articles
    
    async def _collect_from_guardian(
        self, 
        categories: List[str], 
        max_articles: int
    ) -> List[Dict[str, Any]]:
        """
        Thu thập từ Guardian API
        """
        articles = []
        # Implementation for Guardian API
        # Similar structure to NewsAPI
        return articles
    
    async def _collect_from_rss(self, max_articles: int) -> List[Dict[str, Any]]:
        """
        Thu thập từ RSS feeds
        """
        import feedparser
        
        rss_feeds = [
            'http://feeds.bbci.co.uk/news/rss.xml',
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.reuters.com/reuters/topNews'
        ]
        
        articles = []
        articles_per_feed = max_articles // len(rss_feeds)
        
        async with aiohttp.ClientSession() as session:
            for feed_url in rss_feeds:
                try:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            feed_content = await response.text()
                            feed = feedparser.parse(feed_content)
                            
                            for entry in feed.entries[:articles_per_feed]:
                                articles.append({
                                    'title': entry.title,
                                    'content': entry.get('description', ''),
                                    'url': entry.link,
                                    'source': feed.feed.get('title', 'RSS Feed'),
                                    'author': entry.get('author'),
                                    'published_at': self._parse_datetime(
                                        entry.get('published', '')
                                    ),
                                    'category': 'general',
                                    'language': 'en'
                                })
                
                except Exception as e:
                    logger.error(f"Error collecting from RSS {feed_url}: {e}")
                
                await asyncio.sleep(0.5)  # Rate limiting
        
        return articles
    
    async def _process_and_save_articles(
        self, 
        articles: List[Dict[str, Any]]
    ) -> int:
        """
        Xử lý và lưu bài viết vào database
        """
        db = SessionLocal()
        saved_count = 0
        
        try:
            for article_data in articles:
                # Check if article already exists
                existing = db.query(Article).filter(
                    Article.url == article_data['url']
                ).first()
                
                if existing:
                    continue
                
                # Analyze sentiment
                sentiment_data = await self.sentiment_analyzer.analyze_text(
                    article_data.get('content', article_data['title'])
                )
                
                # Create article
                article = Article(
                    title=article_data['title'],
                    content=article_data.get('content'),
                    url=article_data['url'],
                    source=article_data['source'],
                    author=article_data.get('author'),
                    published_at=article_data.get('published_at'),
                    category=article_data.get('category'),
                    language=article_data.get('language', 'en'),
                    sentiment_score=sentiment_data.get('score'),
                    sentiment_label=sentiment_data.get('label'),
                    sentiment_confidence=sentiment_data.get('confidence'),
                    emotional_intensity=sentiment_data.get('intensity', 0)
                )
                
                db.add(article)
                saved_count += 1
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            db.rollback()
        finally:
            db.close()
        
        return saved_count
    
    def _is_valid_article(self, article_data: Dict[str, Any]) -> bool:
        """
        Kiểm tra tính hợp lệ của bài viết
        """
        required_fields = ['title', 'url', 'source']
        return all(
            article_data.get(field) and 
            article_data[field] != '[Removed]' 
            for field in required_fields
        )
    
    def _parse_datetime(self, date_string: str) -> Optional[datetime]:
        """
        Chuyển đổi string thành datetime
        """
        if not date_string:
            return None
        
        try:
            from dateutil import parser
            return parser.parse(date_string)
        except Exception:
            return None
```

---

## 7. Sentiment Analysis Engine

### Bước 7.1: Tạo file services/sentiment_analyzer.py
```python
import asyncio
import logging
from typing import Dict, Any, Optional, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """
        Load các model sentiment analysis
        """
        try:
            # English model - RoBERTa for Twitter sentiment
            self.models['en'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Vietnamese model (if needed)
            # self.models['vi'] = pipeline(
            #     "sentiment-analysis", 
            #     model="uitnlp/vietnamese-sentiment",
            #     return_all_scores=True
            # )
            
            logger.info("Sentiment analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            self.models['en'] = None
    
    async def analyze_text(
        self, 
        text: str, 
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Phân tích cảm xúc của văn bản
        """
        if not text or not text.strip():
            return self._default_sentiment()
        
        try:
            # Get appropriate model
            model = self.models.get(language, self.models.get('en'))
            if not model:
                return self._default_sentiment()
            
            # Preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Run sentiment analysis
            results = model(cleaned_text)
            
            # Process results
            sentiment_data = self._process_results(results)
            
            # Calculate additional metrics
            sentiment_data['intensity'] = self._calculate_intensity(results)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._default_sentiment()
    
    async def analyze_articles_batch(
        self, 
        articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Phân tích cảm xúc cho nhiều bài viết cùng lúc
        """
        results = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            batch_results = await asyncio.gather(
                *[self.analyze_text(
                    article.get('content', article.get('title', '')),
                    article.get('language', 'en')
                ) for article in batch],
                return_exceptions=True
            )
            
            results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (model limits)
        max_length = 500
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def _process_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Xử lý kết quả từ model
        """
        if not results or not results[0]:
            return self._default_sentiment()
        
        # Get the prediction with highest score
        best_prediction = max(results[0], key=lambda x: x['score'])
        
        # Map labels to standard format
        label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive',
            'NEGATIVE': 'negative',
            'NEUTRAL': 'neutral',
            'POSITIVE': 'positive'
        }
        
        label = label_mapping.get(
            best_prediction['label'].upper(), 
            'neutral'
        )
        
        # Convert to score (-1 to 1 scale)
        score = self._convert_to_score(label, best_prediction['score'])
        
        return {
            'label': label,
            'score': score,
            'confidence': best_prediction['score'],
            'raw_results': results[0]
        }
    
    def _convert_to_score(self, label: str, confidence: float) -> float:
        """
        Chuyển đổi label và confidence thành score từ -1 đến 1
        """
        if label == 'positive':
            return confidence
        elif label == 'negative':
            return -confidence
        else:  # neutral
            return 0.0
    
    def _calculate_intensity(self, results: List[Dict[str, Any]]) -> float:
        """
        Tính toán độ mạnh của cảm xúc
        """
        if not results or not results[0]:
            return 0.0
        
        # Calculate as max confidence among all sentiments
        max_confidence = max(pred['score'] for pred in results[0])
        return max_confidence
    
    def _default_sentiment(self) -> Dict[str, Any]:
        """
        Trả về sentiment mặc định khi có lỗi
        """
        return {
            'label': 'neutral',
            'score': 0.0,
            'confidence': 0.0,
            'intensity': 0.0,
            'raw_results': []
        }

# Advanced emotion analysis (optional)
class EmotionAnalyzer:
    def __init__(self):
        try:
            self.emotion_model = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            self.emotion_model = None
    
    async def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """
        Phân tích cảm xúc chi tiết (joy, sadness, anger, fear, etc.)
        """
        if not self.emotion_model or not text:
            return {}
        
        try:
            results = self.emotion_model(text)
            emotions = {pred['label']: pred['score'] for pred in results[0]}
            return emotions
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return {}
```

Đây là phần đầu của hướng dẫn implementation backend. Bạn có muốn tôi tiếp tục với các phần còn lại như Search Engine, Caching, Authentication không?


