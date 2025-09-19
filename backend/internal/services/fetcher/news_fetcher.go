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

// NewsFetcher handles fetching news articles from external APIs
type NewsFetcher struct {
	apiKey  string
	baseURL string
}

// NewNewsFetcher creates a new instance of the news fetcher
func NewNewsFetcher() *NewsFetcher {
	// Check for API key in environment variables
	apiKey := os.Getenv("NEWS_API_KEY")
	if apiKey == "" {
		apiKey = "demo" // Fallback to demo mode if no API key provided
	}

	return &NewsFetcher{
		apiKey:  apiKey,
		baseURL: "https://newsapi.org/v2",
	}
}

// NewsAPIResponse represents the response from the News API
type NewsAPIResponse struct {
	Status       string `json:"status"`
	TotalResults int    `json:"totalResults"`
	Articles     []struct {
		Source struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"source"`
		Author      string    `json:"author"`
		Title       string    `json:"title"`
		Description string    `json:"description"`
		URL         string    `json:"url"`
		URLToImage  string    `json:"urlToImage"`
		PublishedAt time.Time `json:"publishedAt"`
		Content     string    `json:"content"`
	} `json:"articles"`
}

// FetchNewsOptions represents options for fetching news
type FetchNewsOptions struct {
	Country  string
	Category string
	Query    string
	PageSize int
	Page     int
}

// FetchTopHeadlines fetches top headlines from the News API
func (f *NewsFetcher) FetchTopHeadlines(options FetchNewsOptions) ([]models.NewsInput, error) {
	// Build query parameters
	params := []string{}

	if options.Country != "" {
		params = append(params, fmt.Sprintf("country=%s", options.Country))
	} else {
		params = append(params, "country=gh") // Default to Ghana
	}

	if options.Category != "" {
		params = append(params, fmt.Sprintf("category=%s", options.Category))
	}

	if options.Query != "" {
		params = append(params, fmt.Sprintf("q=%s", options.Query))
	}

	if options.PageSize > 0 {
		params = append(params, fmt.Sprintf("pageSize=%d", options.PageSize))
	} else {
		params = append(params, "pageSize=20") // Default page size
	}

	if options.Page > 0 {
		params = append(params, fmt.Sprintf("page=%d", options.Page))
	}

	// Build the URL
	url := fmt.Sprintf("%s/top-headlines?%s", f.baseURL, strings.Join(params, "&"))

	// Create the request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("X-Api-Key", f.apiKey)

	// Make the request
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error fetching news: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse the response
	var newsResp NewsAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&newsResp); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	// Map the API response to our models
	var newsInputs []models.NewsInput
	for _, article := range newsResp.Articles {
		// Skip articles with empty title or content
		if article.Title == "" || article.Content == "" {
			continue
		}

		content := article.Content
		if content == "" {
			content = article.Description // Fallback to description if content is empty
		}

		newsInputs = append(newsInputs, models.NewsInput{
			Title:    article.Title,
			Content:  content,
			URL:      article.URL,
			Source:   article.Source.Name,
			Category: options.Category, // Use the requested category
		})
	}

	return newsInputs, nil
}

// FetchEverything searches for articles from the News API
func (f *NewsFetcher) FetchEverything(query string, options FetchNewsOptions) ([]models.NewsInput, error) {
	// Build query parameters
	params := []string{fmt.Sprintf("q=%s", query)}

	if options.PageSize > 0 {
		params = append(params, fmt.Sprintf("pageSize=%d", options.PageSize))
	} else {
		params = append(params, "pageSize=20") // Default page size
	}

	if options.Page > 0 {
		params = append(params, fmt.Sprintf("page=%d", options.Page))
	}

	// Add sort by relevance
	params = append(params, "sortBy=relevancy")

	// Build the URL
	url := fmt.Sprintf("%s/everything?%s", f.baseURL, strings.Join(params, "&"))

	// Create the request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("X-Api-Key", f.apiKey)

	// Make the request
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error fetching news: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse the response
	var newsResp NewsAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&newsResp); err != nil {
		return nil, fmt.Errorf("error parsing response: %w", err)
	}

	// Map the API response to our models
	var newsInputs []models.NewsInput
	for _, article := range newsResp.Articles {
		// Skip articles with empty title or content
		if article.Title == "" || article.Content == "" {
			continue
		}

		content := article.Content
		if content == "" {
			content = article.Description // Fallback to description if content is empty
		}

		// Determine category based on keywords
		category := DetermineCategory(article.Title + " " + content)

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

// DetermineCategory tries to determine the news category based on keywords
func DetermineCategory(text string) string {
	text = strings.ToLower(text)

	// Define category keywords
	categories := map[string][]string{
		"politics":      {"politics", "government", "election", "president", "minister", "parliament", "democratic", "republican"},
		"business":      {"business", "economy", "market", "stock", "trade", "company", "corporate", "finance", "investment"},
		"technology":    {"technology", "tech", "computer", "software", "hardware", "digital", "internet", "app", "smartphone"},
		"health":        {"health", "medical", "doctor", "hospital", "disease", "virus", "medicine", "covid", "vaccine"},
		"sports":        {"sports", "football", "soccer", "basketball", "tennis", "athlete", "game", "tournament", "championship"},
		"entertainment": {"entertainment", "movie", "film", "music", "celebrity", "actor", "actress", "tv", "television"},
		"science":       {"science", "research", "scientist", "study", "discovery", "biology", "physics", "chemistry"},
	}

	// Count keyword matches for each category
	categoryScores := make(map[string]int)
	for category, keywords := range categories {
		for _, keyword := range keywords {
			if strings.Contains(text, keyword) {
				categoryScores[category]++
			}
		}
	}

	// Find the category with the highest score
	var topCategory string
	var topScore int
	for category, score := range categoryScores {
		if score > topScore {
			topScore = score
			topCategory = category
		}
	}

	// Default to "general" if no strong category match
	if topScore < 2 {
		return "general"
	}

	return topCategory
}
