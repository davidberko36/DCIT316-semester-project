# Backend Modifications for News Fetching

To increase the number of news stories fetched and ensure proper filtering for Ghana news, the following changes are recommended:

## 1. Increase Default News Limits

### In `internal/handlers/news_external_handler.go`:
```go
// Update line 32 to increase default limit from 10 to 30
limit, _ := strconv.Atoi(c.DefaultQuery("limit", "30"))

// Also update line 58 to match
limit, _ := strconv.Atoi(c.DefaultQuery("limit", "30"))
```

### In `internal/handlers/news_handler.go`:
```go
// Update line 28 to increase default limit from 20 to 30
limit, _ := strconv.Atoi(c.DefaultQuery("limit", "30"))

// Update line 167 to increase recommendations from 10 to 20
limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))

// Update line 188 to increase fetch limit from 20 to 30
limit, _ := strconv.Atoi(c.DefaultQuery("limit", "30"))
```

## 2. Ensure Ghana News Filter is Applied in Fetchers

### In `internal/services/fetcher/news_fetcher.go`:
Verify that the Ghana country filter is consistently applied:
```go
// Make sure this is set to "gh" for all relevant endpoints
params = append(params, "country=gh") // Default to Ghana
```

### In `internal/services/fetcher/apitube_fetcher.go`:
Verify the country code is properly set:
```go
// Ensure the Ghana filter is properly applied
params = append(params, "source.country.code=gh") // Default to Ghana
```

## 3. Add More Categories for Ghana News

### In `internal/services/fetcher/composite_fetcher.go`:
Consider adding a function to fetch news from multiple categories at once:

```go
// FetchGhanaNewsByCategories fetches news from multiple categories
func (f *CompositeFetcher) FetchGhanaNewsByCategories(categories []string, limitPerCategory int) []models.NewsInput {
    var allArticles []models.NewsInput
    
    for _, category := range categories {
        options := FetchNewsOptions{
            Country:  "gh",
            Category: category,
            PageSize: limitPerCategory,
        }
        
        articles, err := f.FetchTopHeadlines(options)
        if err == nil && len(articles) > 0 {
            allArticles = append(allArticles, articles...)
        }
    }
    
    return allArticles
}
```

### In `internal/handlers/news_external_handler.go`:
Add a new endpoint to fetch news from multiple categories:

```go
// FetchGhanaNewsByCategories fetches Ghana news from multiple categories
func (h *NewsExternalHandler) FetchGhanaNewsByCategories(c *gin.Context) {
    // Default categories if none provided
    categoriesStr := c.DefaultQuery("categories", "general,business,technology,entertainment,sports,health,science")
    categories := strings.Split(categoriesStr, ",")
    limitPerCategory, _ := strconv.Atoi(c.DefaultQuery("limitPerCategory", "10"))
    
    var allArticles []models.NewsInput
    for _, category := range categories {
        articles, err := h.newsFetcher.GetGhanaNews(category, limitPerCategory)
        if err == nil && len(articles) > 0 {
            allArticles = append(allArticles, articles...)
        }
    }
    
    if len(allArticles) == 0 {
        c.JSON(http.StatusOK, gin.H{
            "message":  "No Ghana news articles found",
            "articles": []interface{}{},
        })
        return
    }
    
    c.JSON(http.StatusOK, gin.H{
        "articles": allArticles,
    })
}
```

Add this route in the RegisterRoutes method:
```go
externalGroup.GET("/ghana/categories", h.FetchGhanaNewsByCategories)
```

## 4. Implement Pagination for Frontend Support

To properly support pagination in the frontend, ensure the backend returns total count information:

### In `internal/handlers/news_handler.go`:
```go
// Update the GetNews function to include total count
func (h *NewsHandler) GetNews(c *gin.Context) {
    // Parse query parameters
    category := c.Query("category")
    limit, _ := strconv.Atoi(c.DefaultQuery("limit", "30"))
    offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

    news, total, err := h.newsService.GetNewsWithTotal(category, limit, offset)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }

    c.JSON(http.StatusOK, gin.H{
        "articles": news,
        "totalResults": total,
        "pageSize": limit,
        "offset": offset,
    })
}
```

The corresponding service method in `internal/services/news_service.go` would need to be implemented to return the total count.

These changes will ensure that:
1. More news stories are fetched (30 instead of 10/20)
2. Ghana-specific news is properly filtered
3. Multiple categories can be fetched at once
4. Proper pagination support is added for the frontend