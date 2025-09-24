package services

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// SerpAPIService handles communication with the SerpAPI
type SerpAPIService struct {
	ApiKey string
}

// NewSerpAPIService creates a new SerpAPI service
func NewSerpAPIService() *SerpAPIService {
	apiKey := os.Getenv("SERPAPI_KEY")
	if apiKey == "" {
		// Use a placeholder key to indicate need for proper configuration
		fmt.Println("WARNING: SERPAPI_KEY environment variable not set. You need to get a valid API key from https://serpapi.com/")
		apiKey = "demo_serpapi_key_replace_me"
	}

	return &SerpAPIService{
		ApiKey: apiKey,
	}
}

// SerpAPIResponse represents the response from SerpAPI
type SerpAPIResponse struct {
	SearchMetadata struct {
		ID     string `json:"id"`
		Status string `json:"status"`
	} `json:"search_metadata"`
	NewsResults []struct {
		Position    int    `json:"position"`
		Link        string `json:"link"`
		Title       string `json:"title"`
		Source      string `json:"source"`
		Date        string `json:"date"`
		PublishedAt string `json:"published_at,omitempty"`
		Snippet     string `json:"snippet"`
		Favicon     string `json:"favicon,omitempty"`
		Thumbnail   string `json:"thumbnail,omitempty"`
	} `json:"news_results"`
}

// GetNews fetches news from SerpAPI
func (s *SerpAPIService) GetNews(query, location string, limit int) (*SerpAPIResponse, error) {
	// If using demo key, return mock data
	if s.ApiKey == "demo_serpapi_key_replace_me" {
		return s.getMockNewsData(query, limit), nil
	}

	// Build the SerpAPI URL
	baseURL := "https://serpapi.com/search"

	// Create URL with query parameters
	params := url.Values{}
	params.Add("engine", "google")
	params.Add("q", query)
	params.Add("location", location)
	params.Add("tbm", "nws") // News search
	params.Add("num", fmt.Sprintf("%d", limit))
	params.Add("api_key", s.ApiKey)

	// Create request
	reqURL := fmt.Sprintf("%s?%s", baseURL, params.Encode())
	resp, err := http.Get(reqURL)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch from SerpAPI: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check if the response is successful
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("SerpAPI returned error: %s", string(body))
	}

	// Parse JSON response
	var apiResp SerpAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response: %w", err)
	}

	return &apiResp, nil
}

// getMockNewsData returns mock news data for demo purposes
func (s *SerpAPIService) getMockNewsData(query string, limit int) *SerpAPIResponse {
	// Create a mock response
	mockResp := &SerpAPIResponse{}
	mockResp.SearchMetadata.ID = "mock_id_12345"
	mockResp.SearchMetadata.Status = "Success"

	// Generate some mock news articles
	mockNews := []struct {
		Title     string
		Source    string
		Snippet   string
		Thumbnail string
	}{
		{
			Title:     "Ghana's Economy Shows Strong Growth in Q3 2025",
			Source:    "Ghana Business News",
			Snippet:   "Ghana's economy expanded by 6.8% in the third quarter of 2025, exceeding analysts' expectations and marking the strongest growth since the pandemic.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Ghana+Economy",
		},
		{
			Title:     "New Infrastructure Projects Announced Across Ghana",
			Source:    "Daily Graphic",
			Snippet:   "The government has unveiled plans for major infrastructure developments including roads, bridges and digital connectivity in rural areas.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Infrastructure",
		},
		{
			Title:     "Ghana's Tech Startup Ecosystem Continues to Expand",
			Source:    "TechCrunch",
			Snippet:   "Ghana's technology sector is experiencing rapid growth with over 50 new startups securing funding in the first half of 2025.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Tech+Startups",
		},
		{
			Title:     "Educational Reforms in Ghana Yield Positive Results",
			Source:    "Ghana Education Service",
			Snippet:   "Recent educational reforms have led to improved literacy rates and higher enrollment in secondary education across the country.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Education",
		},
		{
			Title:     "Ghana's Renewable Energy Initiatives Attract International Investment",
			Source:    "Energy Ghana",
			Snippet:   "Major international investors have committed over $500 million to solar and wind energy projects in Ghana, supporting the country's green energy goals.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Renewable+Energy",
		},
		{
			Title:     "Tourism in Ghana Rebounds to Pre-Pandemic Levels",
			Source:    "Travel Weekly",
			Snippet:   "Ghana's tourism sector has fully recovered from the pandemic downturn, with visitor numbers in 2025 matching 2019 figures.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Tourism",
		},
		{
			Title:     "Ghana's Agricultural Exports Reach Record Highs",
			Source:    "Agri Business Ghana",
			Snippet:   "Exports of cocoa, cashews, and other agricultural products from Ghana have reached all-time highs, boosting foreign exchange earnings.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Agriculture",
		},
		{
			Title:     "Healthcare Improvements in Rural Ghana Show Promising Results",
			Source:    "Ghana Health Service",
			Snippet:   "Initiatives to improve healthcare access in rural areas have led to declining infant mortality rates and better management of chronic diseases.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Healthcare",
		},
		{
			Title:     "Ghana's Cultural Festival Season Attracts Global Attention",
			Source:    "Cultural Heritage Daily",
			Snippet:   "Ghana's vibrant festival season is drawing increased international attention, showcasing the country's rich cultural heritage.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Cultural+Festivals",
		},
		{
			Title:     "Sports Development Program Launched Across Ghana",
			Source:    "Ghana Sports News",
			Snippet:   "A comprehensive sports development program has been launched nationwide to nurture young talent in various sporting disciplines.",
			Thumbnail: "https://via.placeholder.com/300x200?text=Sports",
		},
	}

	// Filter news based on query if provided
	filteredNews := []struct {
		Title     string
		Source    string
		Snippet   string
		Thumbnail string
	}{}

	if query != "" && query != "Ghana news" {
		lowerQuery := strings.ToLower(query)
		for _, news := range mockNews {
			if strings.Contains(strings.ToLower(news.Title), lowerQuery) ||
				strings.Contains(strings.ToLower(news.Snippet), lowerQuery) {
				filteredNews = append(filteredNews, news)
			}
		}
	} else {
		filteredNews = mockNews
	}

	// Limit the number of results
	maxIndex := limit
	if maxIndex > len(filteredNews) {
		maxIndex = len(filteredNews)
	}

	// Add news to the response
	for i := 0; i < maxIndex; i++ {
		mockResp.NewsResults = append(mockResp.NewsResults, struct {
			Position    int    `json:"position"`
			Link        string `json:"link"`
			Title       string `json:"title"`
			Source      string `json:"source"`
			Date        string `json:"date"`
			PublishedAt string `json:"published_at,omitempty"`
			Snippet     string `json:"snippet"`
			Favicon     string `json:"favicon,omitempty"`
			Thumbnail   string `json:"thumbnail,omitempty"`
		}{
			Position:    i + 1,
			Link:        fmt.Sprintf("https://example.com/news/%d", i+1),
			Title:       filteredNews[i].Title,
			Source:      filteredNews[i].Source,
			Date:        time.Now().AddDate(0, 0, -i).Format("01/02/2006"),
			PublishedAt: time.Now().AddDate(0, 0, -i).Format(time.RFC3339),
			Snippet:     filteredNews[i].Snippet,
			Favicon:     "https://via.placeholder.com/16x16",
			Thumbnail:   filteredNews[i].Thumbnail,
		})
	}

	return mockResp
}
