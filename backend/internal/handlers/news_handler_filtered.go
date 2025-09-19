package handlers

import (
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
)

// GetFilteredRecommendations returns recommended news with stronger fake news filtering
func (h *NewsHandler) GetFilteredRecommendations(c *gin.Context) {
	// Get the user ID from the authentication context
	userID, exists := c.Get("userID")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "User not authenticated"})
		return
	}

	// Parse query parameters
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	thresholdStr := c.DefaultQuery("threshold", "0.3") // More stringent default threshold
	threshold, err := strconv.ParseFloat(thresholdStr, 64)
	if err != nil {
		threshold = 0.3 // Default on error
	}

	// Get filtered recommendations
	recommendations, err := h.newsService.GetFilteredRecommendations(userID.(int64), limit, threshold)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, recommendations)
}
