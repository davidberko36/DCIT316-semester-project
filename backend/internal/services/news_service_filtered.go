package services

import (
	"fmt"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// GetFilteredRecommendations returns news recommendations for a user with additional fake news filtering
func (s *NewsServiceImpl) GetFilteredRecommendations(userID int64, limit int, fakeNewsProbabilityThreshold float64) ([]models.News, error) {
	// Get standard recommendations first
	recommendations, err := s.GetRecommendations(userID, limit*2) // Get more than needed to account for filtering
	if err != nil {
		return nil, fmt.Errorf("error getting recommendations: %w", err)
	}

	// Apply stricter fake news filtering
	var filteredRecommendations []models.News
	for _, news := range recommendations {
		if news.FakeProbability < fakeNewsProbabilityThreshold {
			filteredRecommendations = append(filteredRecommendations, news)
			if len(filteredRecommendations) >= limit {
				break
			}
		}
	}

	return filteredRecommendations, nil
}
