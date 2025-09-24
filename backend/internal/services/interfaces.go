package services

import (
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// UserService defines the interface for user-related operations
type UserService interface {
	// User management
	CreateUser(username, email, password string) (*models.User, string, error)
	AuthenticateUser(username, password string) (*models.User, string, error)
	GetUserByID(id int64) (*models.User, error)
	UpdateUserProfile(id int64, profile models.ProfileUpdate) (*models.User, error)
}

// NewsService defines the interface for news-related operations
type NewsService interface {
	// News management
	GetNews(category string, limit, offset int) ([]models.News, error)
	GetNewsById(id interface{}) (*models.News, error)
	CreateNews(input models.NewsInput) (*models.News, error)
	UpdateNews(id int64, input models.NewsInput) (*models.News, error)
	DeleteNews(id int64) error

	// User activity
	LogActivity(userID int64, newsID interface{}, activityType string) (*models.UserActivity, error)
	GetUserActivity(userID int64, limit, offset int) ([]models.UserActivity, error)

	// ML-related operations
	DetectFakeNews(title, content string) (map[string]interface{}, error)
	GetRecommendations(userID int64, limit int) ([]models.News, error)
	GetFilteredRecommendations(userID int64, limit int, fakeNewsProbabilityThreshold float64) ([]models.News, error)

	// External news fetching
	FetchAndStoreNews(country, category string, limit int) ([]models.News, error)
	ProcessArticle(input models.NewsInput) (*models.News, error)
	DeleteOldNews(age time.Duration) (int64, error)
}
