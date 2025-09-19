# News Fetcher Implementation

## Overview
We've implemented a news fetcher feature that allows the backend to fetch news articles from external sources and store them in the database. This feature is essential for keeping the news recommendation system up-to-date with the latest news articles.

## Implementation Details

### 1. News Fetcher Package
We created a news fetcher package in `backend/internal/services/fetcher/news_fetcher.go` that handles the communication with the external news API. The key components include:

- `NewsFetcher` struct that manages the API key and base URL
- `FetchTopHeadlines` method to fetch top headlines for a specific country and category
- `FetchEverything` method to search for news articles based on a query
- Helper functions to process and categorize the fetched articles

### 2. NewsService Interface Update
We updated the `NewsService` interface in `backend/internal/services/interfaces.go` to include a new method:

```go
// External news fetching
FetchAndStoreNews(country, category string, limit int) ([]models.News, error)
```

### 3. NewsService Implementation
We implemented the `FetchAndStoreNews` method in `backend/internal/services/fetch_news.go` to:

- Create a news fetcher instance
- Fetch top headlines based on the provided parameters
- Process each article through the fake news detector
- Store the articles in the database
- Return the successfully stored articles

### 4. API Endpoint
We added a new API endpoint in `backend/internal/handlers/news_handler.go` and updated the routes in `backend/internal/handlers/routes.go`:

```go
// FetchExternalNews fetches news from external API and stores them in the database
func (h *NewsHandler) FetchExternalNews(c *gin.Context) {
    // Parse query parameters
    country := c.DefaultQuery("country", "gh") // Default to Ghana
    category := c.Query("category")            // Optional category filter
    limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))

    // Call the service to fetch and store news
    news, err := h.newsService.FetchAndStoreNews(country, category, limit)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("Failed to fetch news: %v", err)})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "message": fmt.Sprintf("Successfully fetched and stored %d news articles", len(news)),
        "news":    news,
    })
}
```

### 5. Configuration
We added a `NEWS_API_KEY` environment variable to the `.env` file to store the API key for the news API.

## Testing the Feature
Once the server is running, you can test the feature with the following curl command:

```bash
curl -X GET "http://localhost:8080/api/news/fetch?country=gh&category=general&limit=10"
```

Query parameters:
- `country`: Two-letter country code (default: "gh" for Ghana)
- `category`: News category (optional, e.g., "general", "business", "technology")
- `limit`: Maximum number of articles to fetch (default: 20)

## Next Steps
1. Consider adding authentication to the fetch endpoint
2. Implement a scheduler to periodically fetch new articles
3. Add more sophisticated filtering options for the fetch endpoint
4. Enhance the categorization logic to better classify articles