package handlers_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/handlers"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services/ml/mocks"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// TestIntegratedML tests the integration of fake news detection and recommendation
func TestIntegratedML(t *testing.T) {
	// Set up mock services
	mockNewsService := new(mocks.MockNewsService)
	mockFakeNewsDetector := new(mocks.MockFakeNewsDetector)
	mockRecommendationEngine := new(mocks.MockRecommendationEngine)

	// Set up handler
	newsHandler := handlers.NewNewsHandler(mockNewsService)
	authMiddleware := middleware.NewAuthMiddleware()

	// Set up router
	gin.SetMode(gin.TestMode)
	router := handlers.SetupRouter(nil, newsHandler, authMiddleware, nil)

	// Test 1: Detect fake news
	t.Run("DetectFakeNews", func(t *testing.T) {
		// Setup mock behavior
		mockNewsService.On("DetectFakeNews", "Test Title", "Test Content").Return(map[string]interface{}{
			"is_fake":        true,
			"probability":    0.8,
			"title":          "Test Title",
			"threshold":      0.5,
			"classification": "Likely Fake",
		}, nil)

		// Create request
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("POST", "/api/news/detect", strings.NewReader(`{
			"title": "Test Title",
			"content": "Test Content"
		}`))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(w, req)

		// Assertions
		assert.Equal(t, http.StatusOK, w.Code)

		var response map[string]interface{}
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Equal(t, true, response["is_fake"])
		assert.Equal(t, 0.8, response["probability"])
		assert.Equal(t, "Likely Fake", response["classification"])
	})

	// Test 2: Get filtered recommendations
	t.Run("GetFilteredRecommendations", func(t *testing.T) {
		// Setup test user in context
		userId := int64(1)

		// Create mock recommendations
		mockRecommendations := []models.News{
			{
				ID:              1,
				Title:           "Safe News",
				Content:         "Safe content",
				IsFake:          false,
				FakeProbability: 0.1,
			},
			{
				ID:              2,
				Title:           "Suspicious News",
				Content:         "Suspicious content",
				IsFake:          false,
				FakeProbability: 0.4,
			},
		}

		// Setup mock behavior
		mockNewsService.On("GetFilteredRecommendations", userId, 10, 0.3).Return(mockRecommendations, nil)

		// Create authenticated request
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("GET", "/api/user/recommendations/filtered?threshold=0.3", nil)

		// Set user in context (simulating authenticated request)
		ctx, _ := gin.CreateTestContext(w)
		ctx.Set("userID", userId)
		ctx.Request = req

		// Call handler directly since we need to set context
		newsHandler.GetFilteredRecommendations(ctx)

		// Assertions
		assert.Equal(t, http.StatusOK, w.Code)

		var response []models.News
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		assert.Len(t, response, 2)
		assert.Equal(t, "Safe News", response[0].Title)
		assert.Equal(t, 0.1, response[0].FakeProbability)
	})
}
