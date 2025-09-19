// Package services provides news-related services
// This file contains external news processing functionality
package services

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// ProcessArticle processes a news article through the fake news detector and stores it
func (s *NewsServiceImpl) ProcessArticle(input models.NewsInput) (*models.News, error) {
	// Get the threshold from environment
	threshold := 0.5 // Default threshold
	thresholdStr := os.Getenv("FAKE_NEWS_THRESHOLD")
	if thresholdStr != "" {
		if val, err := strconv.ParseFloat(thresholdStr, 64); err == nil {
			threshold = val
		}
	}

	// Detect if the news is fake
	isFake, probability, err := s.fakeNewsDetector.IsFake(input.Title, input.Content, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to predict fake news: %w", err)
	}

	// Create the news object
	news := &models.News{
		Title:           input.Title,
		Content:         input.Content,
		URL:             input.URL,
		Source:          input.Source,
		Category:        input.Category,
		IsFake:          isFake,
		FakeProbability: probability,
		CreatedAt:       time.Now(),
	}

	// Insert into database
	query := `
		INSERT INTO news (title, content, url, source, category, is_fake, fake_probability, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`

	res, err := s.db.Exec(
		query,
		news.Title,
		news.Content,
		news.URL,
		news.Source,
		news.Category,
		news.IsFake,
		news.FakeProbability,
		news.CreatedAt.Format("2006-01-02 15:04:05"),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to insert news: %w", err)
	}

	// Get the ID
	id, err := res.LastInsertId()
	if err != nil {
		return nil, fmt.Errorf("failed to get last insert ID: %w", err)
	}

	news.ID = id
	return news, nil
}
