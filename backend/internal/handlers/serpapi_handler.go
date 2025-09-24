package handlers

import (
	"net/http"
	"strconv"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services"
	"github.com/gin-gonic/gin"
)

// SerpAPIHandler handles requests related to SerpAPI news
type SerpAPIHandler struct {
	serpAPIService *services.SerpAPIService
}

// NewSerpAPIHandler creates a new SerpAPI handler
func NewSerpAPIHandler() *SerpAPIHandler {
	return &SerpAPIHandler{
		serpAPIService: services.NewSerpAPIService(),
	}
}

// GetSerpAPINews handles fetching news from SerpAPI
func (h *SerpAPIHandler) GetSerpAPINews(c *gin.Context) {
	// Get query parameters
	query := c.DefaultQuery("query", "Ghana news")
	location := c.DefaultQuery("location", "Ghana")
	limitStr := c.DefaultQuery("limit", "20") // Default to 20 stories instead of 10

	// Parse limit
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 20 // Default to 20 if parsing fails
	}

	// Fetch news from SerpAPI
	resp, err := h.serpAPIService.GetNews(query, location, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to fetch news from SerpAPI",
			"details": err.Error(),
		})
		return
	}

	// Return the response
	c.JSON(http.StatusOK, resp)
}

// GetSerpAPINewsById handles fetching a specific news article from SerpAPI by ID
func (h *SerpAPIHandler) GetSerpAPINewsById(c *gin.Context) {
	// Get the ID from the URL parameter
	id := c.Param("id")

	// Parse position from ID (format: serpapi-{position})
	positionStr := ""
	for i := 0; i < len(id); i++ {
		if id[i] >= '0' && id[i] <= '9' {
			positionStr = id[i:]
			break
		}
	}

	if positionStr == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid SerpAPI article ID format",
		})
		return
	}

	position, err := strconv.Atoi(positionStr)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Invalid SerpAPI article ID",
		})
		return
	}

	// Fetch news from SerpAPI (we'll get all and then filter)
	resp, err := h.serpAPIService.GetNews("Ghana news", "Ghana", 20) // Get enough articles to find the one we want
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to fetch news from SerpAPI",
			"details": err.Error(),
		})
		return
	}

	// Find the article with the matching position
	var article *struct {
		Position    int    `json:"position"`
		Link        string `json:"link"`
		Title       string `json:"title"`
		Source      string `json:"source"`
		Date        string `json:"date"`
		PublishedAt string `json:"published_at,omitempty"`
		Snippet     string `json:"snippet"`
		Favicon     string `json:"favicon,omitempty"`
		Thumbnail   string `json:"thumbnail,omitempty"`
	}

	for i, a := range resp.NewsResults {
		if a.Position == position {
			article = &resp.NewsResults[i]
			break
		}
	}

	if article == nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": "SerpAPI article not found",
		})
		return
	}

	// Convert to a standard news article format
	result := gin.H{
		"id":              id,
		"title":           article.Title,
		"content":         article.Snippet,
		"description":     article.Snippet,
		"url":             article.Link,
		"source":          article.Source,
		"publishedAt":     article.PublishedAt,
		"date":            article.Date,
		"image":           article.Thumbnail,
		"category":        "news",
		"fakeProbability": 0.5, // Default value
	}

	c.JSON(http.StatusOK, result)
}
