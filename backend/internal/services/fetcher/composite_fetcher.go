package fetcher

import (
	"log"
	"os"
	"strings"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// CompositeFetcher uses multiple news fetchers with fallback capability
type CompositeFetcher struct {
	primary   NewsFetcherInterface
	secondary NewsFetcherInterface
	preferAPI string // which API to try first
}

// NewsFetcherInterface defines the interface for news fetchers
type NewsFetcherInterface interface {
	FetchTopHeadlines(options FetchNewsOptions) ([]models.NewsInput, error)
	FetchEverything(query string, options FetchNewsOptions) ([]models.NewsInput, error)
}

// NewCompositeFetcher creates a new composite fetcher
func NewCompositeFetcher() *CompositeFetcher {
	// Determine which API to use as primary based on environment variables
	preferAPI := "apitube" // default to APITube
	if os.Getenv("PREFER_NEWSAPI") != "" {
		preferAPI = "newsapi"
	}

	// Create both fetchers
	apiTubeFetcher := NewAPITubeFetcher()
	newsAPIFetcher := NewNewsFetcher()

	var primary, secondary NewsFetcherInterface

	if preferAPI == "apitube" {
		primary = apiTubeFetcher
		secondary = newsAPIFetcher
	} else {
		primary = newsAPIFetcher
		secondary = apiTubeFetcher
	}

	return &CompositeFetcher{
		primary:   primary,
		secondary: secondary,
		preferAPI: preferAPI,
	}
}

// FetchTopHeadlines fetches top headlines with fallback
func (f *CompositeFetcher) FetchTopHeadlines(options FetchNewsOptions) ([]models.NewsInput, error) {
	// Try primary fetcher first
	articles, err := f.primary.FetchTopHeadlines(options)
	if err == nil && len(articles) > 0 {
		log.Printf("Successfully fetched %d articles from primary news source", len(articles))
		return articles, nil
	}

	if err != nil {
		log.Printf("Error fetching from primary source: %v. Trying secondary source...", err)
	} else {
		log.Printf("Primary source returned 0 articles. Trying secondary source...")
	}

	// Try secondary fetcher as fallback
	articles, err = f.secondary.FetchTopHeadlines(options)
	if err == nil && len(articles) > 0 {
		log.Printf("Successfully fetched %d articles from secondary news source", len(articles))
		return articles, nil
	}

	if err != nil {
		log.Printf("Error fetching from secondary source: %v", err)
		return nil, err
	}

	log.Printf("Both sources returned 0 articles")
	return articles, nil
}

// FetchEverything searches for articles with fallback
func (f *CompositeFetcher) FetchEverything(query string, options FetchNewsOptions) ([]models.NewsInput, error) {
	// Try primary fetcher first
	articles, err := f.primary.FetchEverything(query, options)
	if err == nil && len(articles) > 0 {
		log.Printf("Successfully fetched %d articles from primary news source", len(articles))
		return articles, nil
	}

	if err != nil {
		log.Printf("Error fetching from primary source: %v. Trying secondary source...", err)
	} else {
		log.Printf("Primary source returned 0 articles. Trying secondary source...")
	}

	// Try secondary fetcher as fallback
	articles, err = f.secondary.FetchEverything(query, options)
	if err == nil && len(articles) > 0 {
		log.Printf("Successfully fetched %d articles from secondary news source", len(articles))
		return articles, nil
	}

	if err != nil {
		log.Printf("Error fetching from secondary source: %v", err)
		return nil, err
	}

	log.Printf("Both sources returned 0 articles")
	return articles, nil
}

// GetGhanaNews is a convenience method specifically for Ghana news
func (f *CompositeFetcher) GetGhanaNews(category string, limit int) ([]models.NewsInput, error) {
	options := FetchNewsOptions{
		Country:  "gh",
		Category: category,
		PageSize: limit,
	}

	// Try to get Ghana-specific headlines
	articles, err := f.FetchTopHeadlines(options)
	if err == nil && len(articles) > 0 {
		return articles, nil
	}

	// If that doesn't work, try a broader search with Ghana keywords
	log.Printf("Trying alternate approach for Ghana news...")

	// Use a Ghana-specific query
	ghanaQuery := "Ghana"
	if category != "" {
		// Add category to the query for more targeted results
		ghanaQuery = "Ghana " + strings.Title(category)
	}

	return f.FetchEverything(ghanaQuery, options)
}
