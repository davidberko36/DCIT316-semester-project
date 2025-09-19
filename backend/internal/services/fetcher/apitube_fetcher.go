package fetcher

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
)

// APITubeFetcher handles fetching news articles from the APITube API
type APITubeFetcher struct {
	apiKey  string
	baseURL string
}

// NewAPITubeFetcher creates a new instance of the APITube fetcher
func NewAPITubeFetcher() *APITubeFetcher {
	// Check for API key in environment variables
	apiKey := os.Getenv("APITUBE_API_KEY")
	if apiKey == "" {
		// If no APITube key, try to use NEWS_API_KEY as fallback
		apiKey = os.Getenv("NEWS_API_KEY")
		if apiKey == "" {
			apiKey = "demo" // Fallback to demo mode if no API key provided
		}
	}

	return &APITubeFetcher{
		apiKey:  apiKey,
		baseURL: "https://api.apitube.io/v1/news",
	}
}

// APITubeResponse represents the response structure from APITube
type APITubeResponse struct {
	Meta     APITubeMeta      `json:"meta"`
	Articles []APITubeArticle `json:"articles"`
}

type APITubeMeta struct {
	Found      int    `json:"found"`
	Returned   int    `json:"returned"`
	Limit      int    `json:"limit"`
	Page       int    `json:"page"`
	TotalPages int    `json:"total_pages"`
	Status     string `json:"status"`
}

type APITubeArticle struct {
	ID          string          `json:"id"`
	Title       string          `json:"title"`
	Description string          `json:"description"`
	Content     string          `json:"content"`
	URL         string          `json:"url"`
	Source      APITubeSource   `json:"source"`
	PublishedAt string          `json:"published_at"`
	Image       string          `json:"image"`
	Language    string          `json:"language"`
	Category    APITubeCategory `json:"category"`
}

type APITubeSource struct {
	Name   string `json:"name"`
	Domain string `json:"domain"`
}

type APITubeCategory struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// FetchTopHeadlines fetches top headlines from the APITube API
func (f *APITubeFetcher) FetchTopHeadlines(options FetchNewsOptions) ([]models.NewsInput, error) {
	// Build query parameters
	params := []string{}

	if options.Country != "" {
		params = append(params, fmt.Sprintf("source.country.code=%s", options.Country))
	} else {
		params = append(params, "source.country.code=gh") // Default to Ghana
	}

	// Map the category to APITube format if provided
	if options.Category != "" {
		apiTubeCategory := convertCategoryToAPITubeFormat(options.Category)
		if apiTubeCategory != "" {
			params = append(params, fmt.Sprintf("category.id=%s", apiTubeCategory))
		}
	}

	if options.Query != "" {
		params = append(params, fmt.Sprintf("title=%s", options.Query))
	}

	if options.PageSize > 0 {
		params = append(params, fmt.Sprintf("per_page=%d", options.PageSize))
	} else {
		params = append(params, "per_page=20") // Default page size
	}

	if options.Page > 0 {
		params = append(params, fmt.Sprintf("page=%d", options.Page))
	}

	// Add language filter for English content
	params = append(params, "language.code=en")

	// Add API key
	params = append(params, fmt.Sprintf("api_key=%s", f.apiKey))

	// Build the URL
	url := fmt.Sprintf("%s/top-headlines?%s", f.baseURL, strings.Join(params, "&"))

	// Make the request
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("error fetching news from APITube: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("APITube API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse the response
	var apiTubeResp APITubeResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiTubeResp); err != nil {
		return nil, fmt.Errorf("error parsing APITube response: %w", err)
	}

	// Map the API response to our models
	var newsInputs []models.NewsInput
	for _, article := range apiTubeResp.Articles {
		// Skip articles with empty title or content
		if article.Title == "" {
			continue
		}

		content := article.Content
		if content == "" {
			content = article.Description // Fallback to description if content is empty
		}

		// If still empty, skip
		if content == "" {
			continue
		}

		category := options.Category
		if category == "" {
			// If no category was specified, use the one from APITube or determine it
			if article.Category.Name != "" {
				category = mapAPITubeCategoryToInternal(article.Category.Name)
			} else {
				category = DetermineCategory(article.Title + " " + content)
			}
		}

		newsInputs = append(newsInputs, models.NewsInput{
			Title:    article.Title,
			Content:  content,
			URL:      article.URL,
			Source:   article.Source.Name,
			Category: category,
		})
	}

	return newsInputs, nil
}

// FetchEverything searches for articles from the APITube API
func (f *APITubeFetcher) FetchEverything(query string, options FetchNewsOptions) ([]models.NewsInput, error) {
	// Build query parameters
	params := []string{}

	if query != "" {
		params = append(params, fmt.Sprintf("title=%s", query))
	}

	if options.Country != "" {
		params = append(params, fmt.Sprintf("source.country.code=%s", options.Country))
	} else {
		params = append(params, "source.country.code=gh") // Default to Ghana
	}

	if options.PageSize > 0 {
		params = append(params, fmt.Sprintf("per_page=%d", options.PageSize))
	} else {
		params = append(params, "per_page=20") // Default page size
	}

	if options.Page > 0 {
		params = append(params, fmt.Sprintf("page=%d", options.Page))
	}

	// Add language filter for English content
	params = append(params, "language.code=en")

	// Add sort parameters
	params = append(params, "sort.by=published_at")
	params = append(params, "sort.order=desc")

	// Add API key
	params = append(params, fmt.Sprintf("api_key=%s", f.apiKey))

	// Build the URL
	url := fmt.Sprintf("%s/everything?%s", f.baseURL, strings.Join(params, "&"))

	// Make the request
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("error fetching news from APITube: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("APITube API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse the response
	var apiTubeResp APITubeResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiTubeResp); err != nil {
		return nil, fmt.Errorf("error parsing APITube response: %w", err)
	}

	// Map the API response to our models
	var newsInputs []models.NewsInput
	for _, article := range apiTubeResp.Articles {
		// Skip articles with empty title or content
		if article.Title == "" {
			continue
		}

		content := article.Content
		if content == "" {
			content = article.Description // Fallback to description if content is empty
		}

		// If still empty, skip
		if content == "" {
			continue
		}

		// Determine category
		category := DetermineCategory(article.Title + " " + content)
		if article.Category.Name != "" {
			category = mapAPITubeCategoryToInternal(article.Category.Name)
		}

		newsInputs = append(newsInputs, models.NewsInput{
			Title:    article.Title,
			Content:  content,
			URL:      article.URL,
			Source:   article.Source.Name,
			Category: category,
		})
	}

	return newsInputs, nil
}

// Helper function to convert our category names to APITube category IDs
func convertCategoryToAPITubeFormat(category string) string {
	categoryMap := map[string]string{
		"general":       "medtop:11000000", // News/Politics
		"business":      "medtop:04000000", // Economy/Business/Finance
		"technology":    "medtop:13000000", // Science and Technology
		"sports":        "medtop:15000000", // Sport
		"health":        "medtop:07000000", // Health
		"entertainment": "medtop:01000000", // Arts/Culture/Entertainment
		"science":       "medtop:13000000", // Science and Technology (same as tech)
	}

	return categoryMap[category]
}

// Helper function to map APITube categories to our internal categories
func mapAPITubeCategoryToInternal(apiTubeCategory string) string {
	lowerCategory := strings.ToLower(apiTubeCategory)

	if strings.Contains(lowerCategory, "business") || strings.Contains(lowerCategory, "econom") || strings.Contains(lowerCategory, "financ") {
		return "business"
	}
	if strings.Contains(lowerCategory, "tech") || strings.Contains(lowerCategory, "sci") {
		return "technology"
	}
	if strings.Contains(lowerCategory, "sport") || strings.Contains(lowerCategory, "football") || strings.Contains(lowerCategory, "soccer") {
		return "sports"
	}
	if strings.Contains(lowerCategory, "health") || strings.Contains(lowerCategory, "medic") {
		return "health"
	}
	if strings.Contains(lowerCategory, "entertain") || strings.Contains(lowerCategory, "art") || strings.Contains(lowerCategory, "music") || strings.Contains(lowerCategory, "culture") {
		return "entertainment"
	}
	if strings.Contains(lowerCategory, "polit") || strings.Contains(lowerCategory, "govern") {
		return "politics"
	}

	return "general"
}
