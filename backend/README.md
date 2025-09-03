# News Aggregator Backend

Backend API cho hệ thống News Aggregator & Sentiment Analysis được xây dựng với FastAPI, PostgreSQL, Redis và Machine Learning.

## 🚀 Quick Start

```bash
# 1. Setup môi trường
cd news-aggregator/backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Tạo file .env (xem QUICK_START.md)
# 3. Chạy ứng dụng
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 Tài Liệu

| File | Mô Tả |
|------|-------|
| [QUICK_START.md](QUICK_START.md) | Hướng dẫn chạy nhanh |
| [BACKEND_IMPLEMENTATION_GUIDE.md](BACKEND_IMPLEMENTATION_GUIDE.md) | Hướng dẫn implementation chi tiết |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Cấu trúc dự án và giải thích |

## 🏗️ Kiến Trúc

- **FastAPI**: Web framework hiệu suất cao
- **PostgreSQL**: Database chính với full-text search
- **Redis**: Caching và session storage
- **Celery**: Background tasks và scheduling
- **Transformers**: ML models cho sentiment analysis
- **Sentence Transformers**: Semantic search
- **Docker**: Containerization

## 🔧 Tính Năng Chính

### ✅ Đã Setup:
- [x] FastAPI application structure
- [x] Environment configuration (.env)
- [x] Dependencies list (requirements.txt)
- [x] Git ignore configuration
- [x] Project documentation

### 📋 Cần Implement:
- [ ] Database models & migrations
- [ ] News collection từ multiple APIs
- [ ] Sentiment analysis với ML
- [ ] Semantic search engine
- [ ] User authentication & preferences
- [ ] Real-time updates với WebSocket
- [ ] Background task automation
- [ ] Comprehensive testing
- [ ] Docker deployment

## 🗂️ Cấu Trúc API

```
/api/v1/
├── /news/              # Tin tức
│   ├── GET /           # Danh sách tin tức
│   ├── GET /search     # Tìm kiếm
│   ├── GET /trending   # Trending topics
│   └── GET /{id}       # Chi tiết bài viết
├── /auth/              # Authentication
│   ├── POST /register  # Đăng ký
│   ├── POST /login     # Đăng nhập
│   └── GET /me         # User info
├── /sentiment/         # Sentiment analysis
│   └── POST /analyze   # Phân tích cảm xúc
└── /health/            # Health checks
    ├── GET /           # Basic health
    └── GET /detailed   # Detailed health
```

## 🔄 Development Workflow

1. **Đọc tài liệu**: Bắt đầu với `BACKEND_IMPLEMENTATION_GUIDE.md`
2. **Setup environment**: Làm theo `QUICK_START.md`
3. **Implement từng component**: Theo thứ tự trong guide
4. **Testing**: Viết tests cho mỗi feature
5. **Deployment**: Sử dụng Docker setup

## 🐳 Docker Deployment

```bash
# Build và chạy tất cả services
docker-compose -f docker/docker-compose.yml up --build

# Services bao gồm:
# - app: FastAPI application
# - db: PostgreSQL database
# - redis: Redis cache
# - celery: Background worker
# - celery-beat: Task scheduler
```

## 🧪 Testing

```bash
# Chạy all tests
pytest tests/ -v

# Test với coverage
pytest --cov=src tests/

# Test specific module
pytest tests/test_api.py -v
```

## 📊 Monitoring

- **Health checks**: `/api/health/` endpoints
- **Logs**: Structured logging với timestamps
- **Metrics**: Prometheus-compatible metrics
- **Error tracking**: Detailed error responses

## 🔐 Security

- **JWT Authentication**: Secure token-based auth
- **Password hashing**: bcrypt hashing
- **Input validation**: Pydantic schemas
- **Rate limiting**: API rate limiting
- **CORS**: Configurable CORS policies

## 🚀 Performance

- **Caching**: Redis caching cho frequent queries
- **Background processing**: Celery tasks
- **Database optimization**: Indexes và query optimization
- **Connection pooling**: Database connection pooling

## 📝 Environment Variables

Xem file `.env.example` hoặc `QUICK_START.md` để biết danh sách đầy đủ environment variables cần thiết.

## 🤝 Contributing

1. Đọc documentation trước khi bắt đầu
2. Follow coding standards (Black, flake8)
3. Viết tests cho code mới
4. Update documentation khi cần

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs trong thư mục `logs/`
2. Đảm bảo all dependencies đã được cài đặt
3. Kiểm tra file `.env` configuration
4. Tham khảo implementation guide

---

**Powered by FastAPI + PostgreSQL + Redis + AI/ML**
