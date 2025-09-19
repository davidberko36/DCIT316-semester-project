package mocks

import (
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/stretchr/testify/mock"
)

// MockNewsService is a mock implementation of the NewsService interface
type MockNewsService struct {
	mock.Mock
}

// GetNews is a mock implementation
func (m *MockNewsService) GetNews(category string, limit, offset int) ([]models.News, error) {
	args := m.Called(category, limit, offset)
	return args.Get(0).([]models.News), args.Error(1)
}

// GetNewsById is a mock implementation
func (m *MockNewsService) GetNewsById(id int64) (*models.News, error) {
	args := m.Called(id)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*models.News), args.Error(1)
}

// CreateNews is a mock implementation
func (m *MockNewsService) CreateNews(input models.NewsInput) (*models.News, error) {
	args := m.Called(input)
	return args.Get(0).(*models.News), args.Error(1)
}

// UpdateNews is a mock implementation
func (m *MockNewsService) UpdateNews(id int64, input models.NewsInput) (*models.News, error) {
	args := m.Called(id, input)
	return args.Get(0).(*models.News), args.Error(1)
}

// DeleteNews is a mock implementation
func (m *MockNewsService) DeleteNews(id int64) error {
	args := m.Called(id)
	return args.Error(0)
}

// DetectFakeNews is a mock implementation
func (m *MockNewsService) DetectFakeNews(title, content string) (map[string]interface{}, error) {
	args := m.Called(title, content)
	return args.Get(0).(map[string]interface{}), args.Error(1)
}

// LogActivity is a mock implementation
func (m *MockNewsService) LogActivity(userID, newsID int64, activityType string) (*models.UserActivity, error) {
	args := m.Called(userID, newsID, activityType)
	return args.Get(0).(*models.UserActivity), args.Error(1)
}

// GetUserActivity is a mock implementation
func (m *MockNewsService) GetUserActivity(userID int64, limit, offset int) ([]models.UserActivity, error) {
	args := m.Called(userID, limit, offset)
	return args.Get(0).([]models.UserActivity), args.Error(1)
}

// GetRecommendations is a mock implementation
func (m *MockNewsService) GetRecommendations(userID int64, limit int) ([]models.News, error) {
	args := m.Called(userID, limit)
	return args.Get(0).([]models.News), args.Error(1)
}

// GetFilteredRecommendations is a mock implementation
func (m *MockNewsService) GetFilteredRecommendations(userID int64, limit int, fakeNewsProbabilityThreshold float64) ([]models.News, error) {
	args := m.Called(userID, limit, fakeNewsProbabilityThreshold)
	return args.Get(0).([]models.News), args.Error(1)
}

// DeleteOldNews is a mock implementation
func (m *MockNewsService) DeleteOldNews(age time.Duration) (int64, error) {
	args := m.Called(age)
	return args.Get(0).(int64), args.Error(1)
}

// FetchAndStoreNews is a mock implementation
func (m *MockNewsService) FetchAndStoreNews(country, category string, limit int) ([]models.News, error) {
	args := m.Called(country, category, limit)
	return args.Get(0).([]models.News), args.Error(1)
}

// ProcessArticle is a mock implementation
func (m *MockNewsService) ProcessArticle(input models.NewsInput) (*models.News, error) {
	args := m.Called(input)
	return args.Get(0).(*models.News), args.Error(1)
}
