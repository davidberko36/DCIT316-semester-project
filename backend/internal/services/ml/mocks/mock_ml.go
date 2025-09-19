package mocks

import (
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/stretchr/testify/mock"
)

// MockFakeNewsDetector is a mock implementation of the FakeNewsClassifier interface
type MockFakeNewsDetector struct {
	mock.Mock
}

// Classify is a mock implementation
func (m *MockFakeNewsDetector) Classify(title string, content string) (float64, error) {
	args := m.Called(title, content)
	return args.Get(0).(float64), args.Error(1)
}

// IsFake is a mock implementation
func (m *MockFakeNewsDetector) IsFake(title string, content string, threshold float64) (bool, float64, error) {
	args := m.Called(title, content, threshold)
	return args.Bool(0), args.Get(1).(float64), args.Error(2)
}

// MockRecommendationEngine is a mock implementation of the RecommendationProvider interface
type MockRecommendationEngine struct {
	mock.Mock
}

// GetRecommendations is a mock implementation
func (m *MockRecommendationEngine) GetRecommendations(
	user models.User,
	userActivity []models.UserActivity,
	newsArticles []models.News,
	limit int,
) ([]models.News, error) {
	args := m.Called(user, userActivity, newsArticles, limit)
	return args.Get(0).([]models.News), args.Error(1)
}
