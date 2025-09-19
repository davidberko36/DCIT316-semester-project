package services

import (
	"fmt"
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services/fetcher"
)

// FetchAndStoreNews fetches news from external API and stores them in the database
func (s *NewsServiceImpl) FetchAndStoreNews(country, category string, limit int) ([]models.News, error) {
	// Create a composite news fetcher that tries multiple sources with fallback
	newsFetcher := fetcher.NewCompositeFetcher()

	// Set up fetch options
	options := fetcher.FetchNewsOptions{
		Country:  country,
		Category: category,
		PageSize: limit,
		Page:     1,
	}

	// Fetch news from API
	newsInputs, err := newsFetcher.FetchTopHeadlines(options)
	if err != nil {
		return nil, fmt.Errorf("error fetching news: %w", err)
	}

	// If no articles found, try a broader search approach
	if len(newsInputs) == 0 {
		// Try a more general search if no headlines found
		if category != "" {
			// Try searching by category
			newsInputs, err = newsFetcher.FetchEverything(category, options)
			if err != nil {
				return nil, fmt.Errorf("error fetching news with category search: %w", err)
			}
		}

		// If still no results or no category specified, try with country name
		if len(newsInputs) == 0 {
			countryQuery := country
			if country == "gh" {
				countryQuery = "Ghana"
			}

			newsInputs, err = newsFetcher.FetchEverything(countryQuery, options)
			if err != nil {
				return nil, fmt.Errorf("error fetching news with country search: %w", err)
			}
		}
	}

	// Get the fake news threshold from environment
	threshold := getThreshold()

	// Process and store each article
	var storedNews []models.News
	for _, input := range newsInputs {
		// Check if article with same title or URL already exists
		var count int
		err := s.db.QueryRow(`
			SELECT COUNT(*) FROM news 
			WHERE title = ? OR (url = ? AND url != '')`,
			input.Title, input.URL).Scan(&count)

		if err != nil {
			return nil, err
		}

		// Skip if article already exists
		if count > 0 {
			continue
		}

		// Detect if the news is fake
		isFake, probability, err := s.fakeNewsDetector.IsFake(input.Title, input.Content, threshold)
		if err != nil {
			// Log the error but continue with default values
			fmt.Printf("Error detecting fake news: %v\n", err)
			isFake = false
			probability = 0
		}

		// Create the news article
		now := time.Now()
		result, err := s.db.Exec(`
			INSERT INTO news (title, content, url, source, category, is_fake, fake_probability, created_at) 
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			input.Title, input.Content, input.URL, input.Source, input.Category, isFake, probability, now)

		if err != nil {
			return nil, fmt.Errorf("error storing news: %w", err)
		}

		// Get the ID of the newly inserted article
		id, err := result.LastInsertId()
		if err != nil {
			return nil, err
		}

		// Add to result list
		storedNews = append(storedNews, models.News{
			ID:              id,
			Title:           input.Title,
			Content:         input.Content,
			URL:             input.URL,
			Source:          input.Source,
			Category:        input.Category,
			IsFake:          isFake,
			FakeProbability: probability,
			CreatedAt:       now,
		})
	}

	return storedNews, nil
}
