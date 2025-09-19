package handlers

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services"
	"github.com/gin-gonic/gin"
)

// NewsHandler handles news-related HTTP requests
type NewsHandler struct {
	newsService services.NewsService
}

// NewNewsHandler creates a new NewsHandler
func NewNewsHandler(newsService services.NewsService) *NewsHandler {
	return &NewsHandler{
		newsService: newsService,
	}
}

// GetNews returns a list of news articles
func (h *NewsHandler) GetNews(c *gin.Context) {
	// Parse query parameters
	category := c.Query("category")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	news, err := h.newsService.GetNews(category, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, news)
}

// GetNewsById returns a specific news article
func (h *NewsHandler) GetNewsById(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	news, err := h.newsService.GetNewsById(id)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "News article not found"})
		return
	}

	c.JSON(http.StatusOK, news)
}

// CreateNews creates a new news article
func (h *NewsHandler) CreateNews(c *gin.Context) {
	var input models.NewsInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	news, err := h.newsService.CreateNews(input)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, news)
}

// UpdateNews updates an existing news article
func (h *NewsHandler) UpdateNews(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	var input models.NewsInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	news, err := h.newsService.UpdateNews(id, input)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, news)
}

// DeleteNews deletes a news article
func (h *NewsHandler) DeleteNews(c *gin.Context) {
	idStr := c.Param("id")
	id, err := strconv.ParseInt(idStr, 10, 64)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	if err := h.newsService.DeleteNews(id); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "News article deleted successfully"})
}

// DetectFakeNews detects if a news article is fake
func (h *NewsHandler) DetectFakeNews(c *gin.Context) {
	var input struct {
		Title   string `json:"title" binding:"required"`
		Content string `json:"content" binding:"required"`
	}

	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	result, err := h.newsService.DetectFakeNews(input.Title, input.Content)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, result)
}

// LogActivity logs user activity with news
func (h *NewsHandler) LogActivity(c *gin.Context) {
	userID, _ := c.Get("userID")
	id, _ := userID.(int64)

	var input models.ActivityInput
	if err := c.ShouldBindJSON(&input); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	activity, err := h.newsService.LogActivity(id, input.NewsID, input.ActivityType)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, activity)
}

// GetRecommendations returns news recommendations for a user
func (h *NewsHandler) GetRecommendations(c *gin.Context) {
	userID, _ := c.Get("userID")
	id, _ := userID.(int64)
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))

	recommendations, err := h.newsService.GetRecommendations(id, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, recommendations)
}

// GetUserActivity returns a user's activity history
func (h *NewsHandler) GetUserActivity(c *gin.Context) {
	userID, _ := c.Get("userID")
	id, _ := userID.(int64)
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	activities, err := h.newsService.GetUserActivity(id, limit, offset)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, activities)
}

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
