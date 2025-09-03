# Backend Project Structure - News Aggregator

## Cấu Trúc Thư Mục Hoàn Chỉnh

```
backend/
├── .env                     # Environment variables
├── .gitignore              ✅ # Git ignore file  
├── requirements.txt        ✅ # Python dependencies
├── BACKEND_IMPLEMENTATION_GUIDE.md ✅ # Chi tiết implementation
├── PROJECT_STRUCTURE.md    
├── README.md               # Project overview
├── 
├── src/                    # Source code chính
│   ├── __init__.py
│   ├── main.py            ✅ # FastAPI entry point
│   │
│   ├── config/            # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py    # App settings từ .env
│   │   └── database.py    # Database configuration
│   │
│   ├── models/            # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── article.py     # Article model với sentiment
│   │   ├── user.py        # User model với preferences
│   │   └── sentiment.py   # Sentiment data model
│   │
│   ├── schemas/           # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── article.py     # Article request/response schemas
│   │   ├── user.py        # User schemas
│   │   └── search.py      # Search request schemas
│   │
│   ├── api/              # API routes
│   │   ├── __init__.py
│   │   ├── deps.py       # Dependencies (auth, db)
│   │   └── v1/           # API version 1
│   │       ├── __init__.py
│   │       ├── news.py    # News endpoints
│   │       ├── search.py  # Search endpoints
│   │       ├── sentiment.py # Sentiment analysis endpoints
│   │       ├── auth.py    # Authentication endpoints
│   │       └── health.py  # Health check endpoints
│   │
│   ├── services/         # Business logic
│   │   ├── __init__.py
│   │   ├── news_collector.py    # Thu thập tin tức từ APIs
│   │   ├── sentiment_analyzer.py # Phân tích cảm xúc
│   │   ├── search_engine.py     # Semantic search
│   │   ├── cache_service.py     # Redis caching
│   │   └── recommendation.py    # Content recommendation
│   │
│   ├── utils/            # Utilities
│   │   ├── __init__.py
│   │   ├── security.py   # JWT, password hashing
│   │   ├── helpers.py    # Helper functions
│   │   ├── validators.py # Data validation
│   │   └── logging.py    # Logging configuration
│   │
│   └── tasks/            # Background tasks
│       ├── __init__.py
│       ├── scheduler.py  # Celery tasks & scheduling
│       └── background_jobs.py # Specific background jobs
│
├── tests/                # Test files
│   ├── __init__.py
│   ├── conftest.py      # Test configuration
│   ├── test_api.py      # API endpoint tests
│   ├── test_services.py # Service layer tests
│   ├── test_models.py   # Model tests
│   └── test_utils.py    # Utility function tests
│
├── migrations/          # Database migrations
│   └── alembic/         # Alembic migration files
│
├── scripts/             # Utility scripts
│   ├── init_db.py      # Database initialization
│   ├── populate_data.py # Sample data creation
│   └── backup_db.py    # Database backup script
│
├── docker/              # Docker configuration
│   ├── Dockerfile      # Application container
│   ├── docker-compose.yml # Multi-service setup
│   └── docker-compose.prod.yml # Production setup
│
└── logs/               # Log files (created at runtime)
    ├── app.log
    ├── celery.log
    └── error.log
```

## Mô Tả Chi Tiết Từng Component

### 1. **Core Configuration (src/config/)**
- `settings.py`: Quản lý tất cả environment variables
- `database.py`: SQLAlchemy setup, connection pooling

### 2. **Data Models (src/models/)**
- `article.py`: Bài viết với sentiment analysis data
- `user.py`: User với preferences cho personalization
- `sentiment.py`: Metadata cho sentiment analysis

### 3. **API Layer (src/api/)**
- `deps.py`: Shared dependencies (auth, database)
- `v1/`: REST API endpoints theo RESTful principles

### 4. **Business Logic (src/services/)**
- `news_collector.py`: Thu thập từ NewsAPI, Guardian, RSS
- `sentiment_analyzer.py`: ML-based sentiment analysis
- `search_engine.py`: Full-text + semantic search
- `cache_service.py`: Redis caching strategies

### 5. **Background Processing (src/tasks/)**
- `scheduler.py`: Celery tasks cho automation
- Automatic news collection, trending updates

### 6. **Testing Strategy (tests/)**
- Unit tests cho từng component
- Integration tests cho API endpoints
- Mock external APIs for reliable testing

### 7. **Deployment (docker/)**
- Multi-container setup: App, PostgreSQL, Redis, Celery
- Production-ready configuration

## Key Features Được Support

### ✅ Đã Implement:
1. **Basic FastAPI setup** với CORS
2. **Environment configuration** (.env support)
3. **Comprehensive .gitignore**
4. **Updated requirements.txt** với all dependencies
5. **Detailed implementation guide**

### 📋 Cần Implement:
1. **Database models** (Article, User)
2. **API endpoints** (News, Search, Auth)
3. **News collection service** (Multiple sources)
4. **Sentiment analysis** (Transformers + ML)
5. **Search engine** (Semantic search)
6. **Caching layer** (Redis)
7. **Authentication** (JWT)
8. **Background tasks** (Celery)
9. **Testing suite**
10. **Docker deployment**

## Thứ Tự Implementation Đề Xuất

1. **Phase 1: Core Setup**
   - Database models
   - Basic API endpoints
   - Simple news collection

2. **Phase 2: ML Features**
   - Sentiment analysis
   - Search engine setup
   - Caching implementation

3. **Phase 3: Advanced Features**
   - User authentication
   - Recommendation system
   - Background task automation

4. **Phase 4: Production Ready**
   - Comprehensive testing
   - Docker deployment
   - Monitoring & logging

## Lệnh Tạo Cấu Trúc

```bash
# Tạo tất cả thư mục cần thiết
mkdir -p src/{config,models,schemas,api/v1,services,utils,tasks}
mkdir -p tests migrations scripts docker logs

# Tạo __init__.py files
touch src/__init__.py
touch src/config/__init__.py
touch src/models/__init__.py
touch src/schemas/__init__.py
touch src/api/__init__.py
touch src/api/v1/__init__.py
touch src/services/__init__.py
touch src/utils/__init__.py
touch src/tasks/__init__.py
touch tests/__init__.py

# Tạo main files
touch src/config/{settings.py,database.py}
touch src/models/{article.py,user.py,sentiment.py}
touch src/schemas/{article.py,user.py,search.py}
touch src/api/{deps.py}
touch src/api/v1/{news.py,search.py,sentiment.py,auth.py,health.py}
touch src/services/{news_collector.py,sentiment_analyzer.py,search_engine.py,cache_service.py}
touch src/utils/{security.py,helpers.py,validators.py,logging.py}
touch src/tasks/{scheduler.py,background_jobs.py}
touch tests/{conftest.py,test_api.py,test_services.py,test_models.py}
touch scripts/{init_db.py,populate_data.py}
touch docker/{Dockerfile,docker-compose.yml}
```

Tham khảo `BACKEND_IMPLEMENTATION_GUIDE.md` để biết chi tiết code cho từng file!
