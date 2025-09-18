package ml

import (
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// FakeNewsClassifier is an interface for fake news detection
type FakeNewsClassifier interface {
	// Classify returns the probability that a news article is fake
	Classify(title string, content string) (float64, error)

	// IsFake determines if an article is fake based on a threshold
	IsFake(title string, content string, threshold float64) (bool, float64, error)
}

// RecommendationProvider is an interface for recommendation generation
type RecommendationProvider interface {
	// GetRecommendations returns a list of recommended news articles for a user
	GetRecommendations(
		user models.User,
		userProfile models.UserProfile,
		newsArticles []models.News,
		userActivity []models.UserActivity,
	) ([]models.News, error)
}
