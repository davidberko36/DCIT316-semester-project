package services

import (
	"database/sql"
	"fmt"
	"hash/fnv"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services/ml"
)

// NewsServiceImpl implements the NewsService interface
type NewsServiceImpl struct {
	db                   *sql.DB
	fakeNewsDetector     *ml.FakeNewsDetector
	recommendationEngine *ml.RecommendationEngine
}

// NewNewsService creates a new news service
func NewNewsService(
	db *sql.DB,
	fakeNewsDetector *ml.FakeNewsDetector,
	recommendationEngine *ml.RecommendationEngine,
) *NewsServiceImpl {
	return &NewsServiceImpl{
		db:                   db,
		fakeNewsDetector:     fakeNewsDetector,
		recommendationEngine: recommendationEngine,
	}
}

// GetNews returns a list of news articles
func (s *NewsServiceImpl) GetNews(category string, limit, offset int) ([]models.News, error) {
	query := "SELECT id, title, content, url, source, category, is_fake, fake_probability, created_at FROM news"
	args := []interface{}{}

	if category != "" {
		query += " WHERE category = ?"
		args = append(args, category)
	}

	query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
	args = append(args, limit, offset)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var news []models.News
	for rows.Next() {
		var n models.News
		if err := rows.Scan(
			&n.ID,
			&n.Title,
			&n.Content,
			&n.URL,
			&n.Source,
			&n.Category,
			&n.IsFake,
			&n.FakeProbability,
			&n.CreatedAt,
		); err != nil {
			return nil, err
		}
		news = append(news, n)
	}

	// If flag is set to include SerpAPI results, fetch and add them
	includeSerpAPI := os.Getenv("INCLUDE_SERPAPI_IN_FEED")
	if includeSerpAPI == "true" && len(news) < limit {
		// Create SerpAPI service
		serpAPIService := NewSerpAPIService()

		// Calculate how many additional news items we need
		remainingItems := limit - len(news)

		// Determine the query based on category
		serpQuery := "Ghana news"
		if category != "" {
			serpQuery = category + " news Ghana"
		}

		// Fetch news from SerpAPI
		serpResults, err := serpAPIService.GetNews(serpQuery, "Ghana", remainingItems)
		if err == nil && serpResults != nil && len(serpResults.NewsResults) > 0 {
			// Convert SerpAPI results to news model
			for _, item := range serpResults.NewsResults {
				// Create a news item
				n := models.News{
					ID:              fmt.Sprintf("serpapi-%d", item.Position),
					Title:           item.Title,
					Content:         item.Snippet,
					URL:             item.Link,
					Source:          item.Source,
					Category:        category,
					IsFake:          false,
					FakeProbability: 0.5, // Default value
				}

				// Parse the timestamp if available
				if item.PublishedAt != "" {
					if t, err := time.Parse(time.RFC3339, item.PublishedAt); err == nil {
						n.CreatedAt = t
					}
				} else if item.Date != "" {
					if t, err := time.Parse("01/02/2006", item.Date); err == nil {
						n.CreatedAt = t
					}
				}

				// If we couldn't parse a date, use current time
				if n.CreatedAt.IsZero() {
					n.CreatedAt = time.Now()
				}

				news = append(news, n)
			}
		}
	}

	return news, nil
}

// GetNewsById returns a specific news article
func (s *NewsServiceImpl) GetNewsById(id interface{}) (*models.News, error) {
	// Check if the ID is a string with "serpapi-" prefix
	if idStr, ok := id.(string); ok && len(idStr) >= 7 && idStr[:7] == "serpapi-" {
		// This is a SerpAPI article, fetch it directly
		parts := strings.Split(idStr, "-")
		positionStr := parts[len(parts)-1] // Get the last part which should be the position
		position, err := strconv.Atoi(positionStr)
		if err != nil {
			return nil, fmt.Errorf("invalid SerpAPI article ID format: %w", err)
		}

		// Determine query based on ID format
		query := "Ghana news"
		if len(parts) > 2 {
			// If format is serpapi-category-position, use the category
			category := parts[1]
			query = fmt.Sprintf("%s news Ghana", category)
		}

		// Create SerpAPI service
		serpAPIService := NewSerpAPIService()

		// Fetch news from SerpAPI with a larger limit to ensure we find our article
		serpResults, err := serpAPIService.GetNews(query, "Ghana", 30) // Get enough articles to find the one we want
		if err != nil {
			return nil, fmt.Errorf("failed to fetch news from SerpAPI: %w", err)
		}

		// Debug log
		log.Printf("Looking for SerpAPI article with position %d in %d results\n", position, len(serpResults.NewsResults))

		// Find the article with the matching position
		for _, item := range serpResults.NewsResults {
			if item.Position == position {
				// Create and return the news item
				n := &models.News{
					ID:              idStr,
					Title:           item.Title,
					Content:         item.Snippet,
					URL:             item.Link,
					Source:          item.Source,
					Category:        "news",
					IsFake:          false,
					FakeProbability: 0.5, // Default value
				}

				// Parse the timestamp if available
				if item.PublishedAt != "" {
					if t, err := time.Parse(time.RFC3339, item.PublishedAt); err == nil {
						n.CreatedAt = t
					}
				} else if item.Date != "" {
					if t, err := time.Parse("01/02/2006", item.Date); err == nil {
						n.CreatedAt = t
					}
				}

				// If we couldn't parse a date, use current time
				if n.CreatedAt.IsZero() {
					n.CreatedAt = time.Now()
				}

				return n, nil
			}
		}

		return nil, fmt.Errorf("SerpAPI article with ID %s not found", idStr)
	}

	// Otherwise, treat it as a regular database ID
	var idInt int64
	switch v := id.(type) {
	case int64:
		idInt = v
	case int:
		idInt = int64(v)
	case string:
		var err error
		idInt, err = strconv.ParseInt(v, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid news ID format: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported ID type: %T", id)
	}
	query := `
		SELECT id, title, content, url, source, category, is_fake, fake_probability, created_at 
		FROM news 
		WHERE id = ?
	`

	var news models.News
	err := s.db.QueryRow(query, idInt).Scan(
		&news.ID,
		&news.Title,
		&news.Content,
		&news.URL,
		&news.Source,
		&news.Category,
		&news.IsFake,
		&news.FakeProbability,
		&news.CreatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("news with ID %d not found", idInt)
		}
		return nil, err
	}

	return &news, nil
}

// CreateNews creates a new news article
func (s *NewsServiceImpl) CreateNews(input models.NewsInput) (*models.News, error) {
	// Check if the news is fake
	isFake, probability, err := s.fakeNewsDetector.IsFake(input.Title, input.Content, getThreshold())
	if err != nil {
		return nil, fmt.Errorf("error detecting fake news: %w", err)
	}

	query := `
		INSERT INTO news (title, content, url, source, category, is_fake, fake_probability, created_at) 
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`

	now := time.Now()
	result, err := s.db.Exec(
		query,
		input.Title,
		input.Content,
		input.URL,
		input.Source,
		input.Category,
		isFake,
		probability,
		now,
	)
	if err != nil {
		return nil, err
	}

	id, err := result.LastInsertId()
	if err != nil {
		return nil, err
	}

	return &models.News{
		ID:              id,
		Title:           input.Title,
		Content:         input.Content,
		URL:             input.URL,
		Source:          input.Source,
		Category:        input.Category,
		IsFake:          isFake,
		FakeProbability: probability,
		CreatedAt:       now,
	}, nil
}

// UpdateNews updates an existing news article
func (s *NewsServiceImpl) UpdateNews(id int64, input models.NewsInput) (*models.News, error) {
	// Check if the news exists
	_, err := s.GetNewsById(id)
	if err != nil {
		return nil, err
	}

	// Check if the news is fake
	isFake, probability, err := s.fakeNewsDetector.IsFake(input.Title, input.Content, getThreshold())
	if err != nil {
		return nil, fmt.Errorf("error detecting fake news: %w", err)
	}

	query := `
		UPDATE news 
		SET title = ?, content = ?, url = ?, source = ?, category = ?, is_fake = ?, fake_probability = ? 
		WHERE id = ?
	`

	_, err = s.db.Exec(
		query,
		input.Title,
		input.Content,
		input.URL,
		input.Source,
		input.Category,
		isFake,
		probability,
		id,
	)
	if err != nil {
		return nil, err
	}

	return &models.News{
		ID:              id,
		Title:           input.Title,
		Content:         input.Content,
		URL:             input.URL,
		Source:          input.Source,
		Category:        input.Category,
		IsFake:          isFake,
		FakeProbability: probability,
	}, nil
}

// DeleteNews deletes a news article
func (s *NewsServiceImpl) DeleteNews(id int64) error {
	// Check if the news exists
	_, err := s.GetNewsById(id)
	if err != nil {
		return err
	}

	query := "DELETE FROM news WHERE id = ?"
	_, err = s.db.Exec(query, id)
	return err
}

// DetectFakeNews detects if a news article is fake
func (s *NewsServiceImpl) DetectFakeNews(title, content string) (map[string]interface{}, error) {
	isFake, probability, err := s.fakeNewsDetector.IsFake(title, content, getThreshold())
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"title":          title,
		"is_fake":        isFake,
		"probability":    probability,
		"threshold":      getThreshold(),
		"classification": getClassification(probability),
	}, nil
}

// LogActivity logs user activity with news
func (s *NewsServiceImpl) LogActivity(userID int64, newsID interface{}, activityType string) (*models.UserActivity, error) {
	// Check if the news exists
	_, err := s.GetNewsById(newsID)
	if err != nil {
		return nil, err
	}

	// Convert newsID to the appropriate type for database storage
	var newsIDValue interface{} = newsID

	// If it's a string ID that starts with "serpapi-", we need special handling
	if idStr, ok := newsID.(string); ok && strings.HasPrefix(idStr, "serpapi-") {
		// For SerpAPI articles, we'll use a negative ID to avoid conflicts with regular IDs
		// Extract the position number and negate it
		posStr := strings.TrimPrefix(idStr, "serpapi-")
		if pos, err := strconv.ParseInt(posStr, 10, 64); err == nil {
			newsIDValue = -pos // Use negative number to indicate SerpAPI ID
		} else {
			// If we can't parse the position, use a hash of the string as the ID
			h := fnv.New64a()
			h.Write([]byte(idStr))
			newsIDValue = -int64(h.Sum64()) // Use negative hash
		}
	}

	query := `
		INSERT INTO user_activity (user_id, news_id, activity_type, created_at) 
		VALUES (?, ?, ?, ?)
	`

	now := time.Now()
	result, err := s.db.Exec(query, userID, newsIDValue, activityType, now)
	if err != nil {
		return nil, err
	}

	id, err := result.LastInsertId()
	if err != nil {
		return nil, err
	}

	return &models.UserActivity{
		ID:           id,
		UserID:       userID,
		NewsID:       newsID,
		ActivityType: activityType,
		CreatedAt:    now,
	}, nil
}

// GetUserActivity returns a user's activity history
func (s *NewsServiceImpl) GetUserActivity(userID int64, limit, offset int) ([]models.UserActivity, error) {
	query := `
		SELECT id, user_id, news_id, activity_type, created_at 
		FROM user_activity 
		WHERE user_id = ? 
		ORDER BY created_at DESC 
		LIMIT ? OFFSET ?
	`

	rows, err := s.db.Query(query, userID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var activities []models.UserActivity
	for rows.Next() {
		var a models.UserActivity
		if err := rows.Scan(&a.ID, &a.UserID, &a.NewsID, &a.ActivityType, &a.CreatedAt); err != nil {
			return nil, err
		}
		activities = append(activities, a)
	}

	return activities, nil
}

// GetRecommendations returns news recommendations for a user
func (s *NewsServiceImpl) GetRecommendations(userID int64, limit int) ([]models.News, error) {
	// Get user information
	var user models.User
	userQuery := `
		SELECT u.id, u.username, u.email, u.created_at, u.updated_at,
			p.id, p.user_id, p.age, p.gender, p.location, p.interests, p.created_at, p.updated_at
		FROM users u
		LEFT JOIN user_profiles p ON u.id = p.user_id
		WHERE u.id = ?
	`

	var profileID, profileUserID sql.NullInt64
	var age sql.NullInt32
	var gender, location, interests sql.NullString
	var profileCreatedAt, profileUpdatedAt sql.NullTime

	err := s.db.QueryRow(userQuery, userID).Scan(
		&user.ID,
		&user.Username,
		&user.Email,
		&user.CreatedAt,
		&user.UpdatedAt,
		&profileID,
		&profileUserID,
		&age,
		&gender,
		&location,
		&interests,
		&profileCreatedAt,
		&profileUpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("error getting user: %w", err)
	}

	// If user has profile, fill it
	if profileID.Valid {
		var agePtr *int
		var genderPtr, locationPtr, interestsPtr *string

		if age.Valid {
			ageInt := int(age.Int32)
			agePtr = &ageInt
		}
		if gender.Valid {
			genderPtr = &gender.String
		}
		if location.Valid {
			locationPtr = &location.String
		}
		if interests.Valid {
			interestsPtr = &interests.String
		}

		user.Profile = &models.UserProfile{
			ID:        profileID.Int64,
			UserID:    profileUserID.Int64,
			Age:       agePtr,
			Gender:    genderPtr,
			Location:  locationPtr,
			Interests: interestsPtr,
			CreatedAt: profileCreatedAt.Time,
			UpdatedAt: profileUpdatedAt.Time,
		}
	}

	// Get user activity
	activityQuery := `
		SELECT id, user_id, news_id, activity_type, created_at
		FROM user_activity
		WHERE user_id = ?
		ORDER BY created_at DESC
		LIMIT 50
	`
	rows, err := s.db.Query(activityQuery, userID)
	if err != nil {
		return nil, fmt.Errorf("error getting user activity: %w", err)
	}
	defer rows.Close()

	var activities []models.UserActivity
	for rows.Next() {
		var a models.UserActivity
		if err := rows.Scan(&a.ID, &a.UserID, &a.NewsID, &a.ActivityType, &a.CreatedAt); err != nil {
			return nil, err
		}
		activities = append(activities, a)
	}

	// Get available news articles
	newsQuery := `
		SELECT id, title, content, url, source, category, is_fake, fake_probability, created_at
		FROM news
		WHERE is_fake = 0
		ORDER BY created_at DESC
		LIMIT 100
	`
	rows, err = s.db.Query(newsQuery)
	if err != nil {
		return nil, fmt.Errorf("error getting news: %w", err)
	}
	defer rows.Close()

	var availableNews []models.News
	for rows.Next() {
		var n models.News
		if err := rows.Scan(
			&n.ID,
			&n.Title,
			&n.Content,
			&n.URL,
			&n.Source,
			&n.Category,
			&n.IsFake,
			&n.FakeProbability,
			&n.CreatedAt,
		); err != nil {
			return nil, err
		}
		availableNews = append(availableNews, n)
	}

	// Get recommendations
	recommendations, err := s.recommendationEngine.GetRecommendations(user, activities, availableNews, limit)
	if err != nil {
		return nil, fmt.Errorf("error getting recommendations: %w", err)
	}

	return recommendations, nil
}

// Helper functions

// getThreshold returns the fake news detection threshold from environment or default
func getThreshold() float64 {
	thresholdStr := os.Getenv("FAKE_NEWS_THRESHOLD")
	if thresholdStr == "" {
		return 0.5 // Default threshold
	}

	threshold, err := strconv.ParseFloat(thresholdStr, 64)
	if err != nil {
		return 0.5 // Default on error
	}

	return threshold
}

// getClassification returns a textual classification based on probability
func getClassification(probability float64) string {
	if probability < 0.3 {
		return "likely legitimate"
	} else if probability < 0.5 {
		return "possibly legitimate"
	} else if probability < 0.7 {
		return "possibly fake"
	} else {
		return "likely fake"
	}
}
