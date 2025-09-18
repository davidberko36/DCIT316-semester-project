package ml

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// RecommendationEngine provides methods for recommending news articles
type RecommendationEngine struct {
	ModelPath string
}

// NewsWithScore represents a news article with a recommendation score
type NewsWithScore struct {
	News  models.News
	Score float64
}

// NewRecommendationEngine loads a recommendation model from an ONNX file
func NewRecommendationEngine(modelPath string) (*RecommendationEngine, error) {
	// Check if the model file exists
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("error resolving model path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", absPath)
	}

	return &RecommendationEngine{
		ModelPath: absPath,
	}, nil
}

// GetRecommendations returns recommended news articles for a user
func (re *RecommendationEngine) GetRecommendations(
	user models.User,
	userHistory []models.UserActivity,
	availableNews []models.News,
	limit int,
) ([]models.News, error) {
	// This is a simplified implementation for testing
	// In a real implementation, we would use ONNX runtime to run the neural network model

	// Score each article based on a simplified algorithm
	scoredNews := make([]NewsWithScore, 0, len(availableNews))

	for _, news := range availableNews {
		score := re.calculateScore(user, userHistory, news)
		scoredNews = append(scoredNews, NewsWithScore{
			News:  news,
			Score: score,
		})
	}

	// Sort by score (highest first)
	sort.Slice(scoredNews, func(i, j int) bool {
		return scoredNews[i].Score > scoredNews[j].Score
	})

	// Take top recommendations
	resultCount := limit
	if resultCount > len(scoredNews) {
		resultCount = len(scoredNews)
	}

	// Extract just the news objects
	result := make([]models.News, resultCount)
	for i := 0; i < resultCount; i++ {
		result[i] = scoredNews[i].News
	}

	return result, nil
}

// calculateScore provides a simplified scoring mechanism for news recommendations
func (re *RecommendationEngine) calculateScore(
	user models.User,
	userHistory []models.UserActivity,
	newsItem models.News,
) float64 {
	score := 0.0

	// Factor 1: Use interests from user profile
	if user.Profile != nil && user.Profile.Interests != nil {
		interests := strings.Split(*user.Profile.Interests, ",")
		for _, interest := range interests {
			interest = strings.TrimSpace(interest)
			if strings.Contains(newsItem.Category, interest) ||
				strings.Contains(newsItem.Title, interest) ||
				strings.Contains(newsItem.Content, interest) {
				score += 0.5
				break
			}
		}
	}

	// Factor 2: Recency (newer articles score higher)
	// This would use the timestamp, but for simplicity we'll use the ID as a proxy
	recencyScore := math.Min(float64(newsItem.ID)*0.01, 0.3)
	score += recencyScore

	// Factor 3: Based on user history (interaction patterns)
	for _, activity := range userHistory {
		if activity.NewsID == newsItem.ID {
			// User has already seen this article, reduce score
			score -= 0.4
			break
		}
	}

	// Ensure score is between 0 and 1
	return math.Max(0, math.Min(1, score))
}

// Helper function to find a news article by ID
func getNewsFromID(newsList []models.News, id int64) *models.News {
	for _, news := range newsList {
		if news.ID == id {
			return &news
		}
	}
	return nil
}
