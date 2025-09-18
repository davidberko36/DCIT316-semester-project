package services

import (
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
	GetNewsById(id int64) (*models.News, error)
	CreateNews(input models.NewsInput) (*models.News, error)
	UpdateNews(id int64, input models.NewsInput) (*models.News, error)
	DeleteNews(id int64) error

	// User activity
	LogActivity(userID, newsID int64, activityType string) (*models.UserActivity, error)
	GetUserActivity(userID int64, limit, offset int) ([]models.UserActivity, error)

	// ML-related operations
	DetectFakeNews(title, content string) (map[string]interface{}, error)
	GetRecommendations(userID int64, limit int) ([]models.News, error)
}
