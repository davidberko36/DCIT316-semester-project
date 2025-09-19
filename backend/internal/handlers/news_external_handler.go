package handlers

import (
	"net/http"
	"strconv"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services/fetcher"
	"github.com/gin-gonic/gin"
)

// NewsExternalHandler handles fetching news from external APIs
type NewsExternalHandler struct {
	newsFetcher *fetcher.CompositeFetcher
	newsService services.NewsService
}

// NewNewsExternalHandler creates a new handler for external news fetching
func NewNewsExternalHandler(newsService services.NewsService) *NewsExternalHandler {
	return &NewsExternalHandler{
		newsFetcher: fetcher.NewCompositeFetcher(),
		newsService: newsService,
	}
}

// FetchGhanaNews fetches Ghana news and returns it directly
func (h *NewsExternalHandler) FetchGhanaNews(c *gin.Context) {
	// Parse query parameters
	category := c.Query("category")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))

	// Call the fetcher to get Ghana news
	articles, err := h.newsFetcher.GetGhanaNews(category, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch Ghana news: " + err.Error(),
		})
		return
	}

	if len(articles) == 0 {
		c.JSON(http.StatusOK, gin.H{
			"message":  "No Ghana news articles found",
			"articles": []interface{}{},
		})
		return
	}

	// Return the articles directly without storing them
	c.JSON(http.StatusOK, gin.H{
		"articles": articles,
	})
}

// FetchAndStoreGhanaNews fetches Ghana news and stores it in the database
func (h *NewsExternalHandler) FetchAndStoreGhanaNews(c *gin.Context) {
	// Parse query parameters
	category := c.Query("category")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))

	// Call the fetcher to get Ghana news
	articles, err := h.newsFetcher.GetGhanaNews(category, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Failed to fetch Ghana news: " + err.Error(),
		})
		return
	}

	if len(articles) == 0 {
		c.JSON(http.StatusOK, gin.H{
			"message": "No Ghana news articles found",
			"count":   0,
		})
		return
	}

	// Store each article
	var savedArticles []interface{}
	for _, article := range articles {
		saved, err := h.newsService.CreateNews(article)
		if err != nil {
			// Log error but continue with other articles
			continue
		}
		savedArticles = append(savedArticles, saved)
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Ghana news fetched and stored",
		"count":   len(savedArticles),
	})
}

// RegisterRoutes registers the external news routes
func (h *NewsExternalHandler) RegisterRoutes(router *gin.Engine) {
	externalGroup := router.Group("/api/external")

	externalGroup.GET("/ghana", h.FetchGhanaNews)
	externalGroup.POST("/ghana/store", h.FetchAndStoreGhanaNews)
}
