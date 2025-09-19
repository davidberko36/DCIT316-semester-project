# Automated News Management System

## Overview
We've implemented an automated news management system that handles periodic fetching of fresh news articles and cleanup of old articles. The system uses goroutines to perform these tasks in the background without blocking the main server.

## Key Components

### 1. News Scheduler
The `NewsScheduler` is responsible for:
- Fetching news articles from various categories at regular intervals (every 10 minutes)
- Cleaning up old news articles (older than 72 hours) to keep the database optimized
- Running these operations concurrently using goroutines

```go
// NewsScheduler handles periodic news fetching and cleanup
type NewsScheduler struct {
    newsService    NewsService
    fetchInterval  time.Duration
    cleanupAge     time.Duration
    stopChan       chan struct{}
    wg             sync.WaitGroup
    isRunning      bool
    fetchCategories []string
    country        string
    articlesPerCat int
}
```

### 2. News Categories
The system fetches news from multiple categories to provide a diverse range of content:
- General
- Business
- Technology
- Sports
- Entertainment
- Health
- Science

Each category is fetched separately to ensure we get a balanced mix of content.

### 3. Cleanup Mechanism
Articles older than 72 hours are automatically removed from the database. This:
- Keeps the database size manageable
- Ensures users see mostly recent news
- Optimizes query performance

## How It Works

### Initialization
When the server starts:
1. The NewsScheduler is created and configured
2. Initial news fetch is triggered immediately
3. Background tasks are started with proper synchronization

### Periodic Operations
The scheduler runs two main tasks:

**News Fetching (every 10 minutes)**
- For each category, fetch articles from the NewsAPI
- Process each article through the fake news detector
- Store articles in the database with proper categorization

**Cleanup (every 30 minutes)**
- Find articles older than 72 hours
- Remove them from the database
- Log the number of articles removed

### Graceful Shutdown
When the server is shutting down:
1. The scheduler is signaled to stop
2. All ongoing operations complete
3. Resources are properly released

## Database Schema
The system utilizes the existing news table which supports categorization:
```sql
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT,
    source TEXT,
    category TEXT,
    is_fake BOOLEAN NOT NULL,
    fake_probability REAL NOT NULL,
    created_at DATETIME NOT NULL
)
```

## Configuration Options
The news scheduler can be configured by modifying:
- `fetchInterval`: How often to fetch news (default: 10 minutes)
- `cleanupAge`: How old articles need to be before removal (default: 72 hours)
- `fetchCategories`: Which categories of news to fetch
- `country`: Default country for news (default: "gh" for Ghana)
- `articlesPerCat`: Number of articles to fetch per category (default: 10)

## Benefits
- Fresh content is automatically added without manual intervention
- Database size remains optimized by removing old content
- Balanced content across different news categories
- Configurable parameters for easy adjustment

## Manual Override
The system still supports manual triggering of news fetching via the API endpoint:
```
GET /api/news/fetch?country=gh&category=business&limit=10
```
This allows for immediate fetching when needed, alongside the automated schedule.