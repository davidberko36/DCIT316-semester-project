package models

import (
	"time"
)

// News represents a news article in the system
type News struct {
	ID              interface{} `json:"id"` // Can be int64 or string
	Title           string      `json:"title"`
	Content         string      `json:"content"`
	URL             string      `json:"url,omitempty"`
	Source          string      `json:"source,omitempty"`
	Category        string      `json:"category,omitempty"`
	IsFake          bool        `json:"is_fake"`
	FakeProbability float64     `json:"fake_probability"`
	CreatedAt       time.Time   `json:"created_at"`
}

// NewsInput is used when adding a new news article
type NewsInput struct {
	Title    string `json:"title" binding:"required"`
	Content  string `json:"content" binding:"required"`
	URL      string `json:"url,omitempty"`
	Source   string `json:"source,omitempty"`
	Category string `json:"category,omitempty"`
}

// UserActivity represents a user's interaction with news
type UserActivity struct {
	ID           int64       `json:"id"`
	UserID       int64       `json:"user_id"`
	NewsID       interface{} `json:"news_id"`
	ActivityType string      `json:"activity_type"` // view, like, dislike, click
	CreatedAt    time.Time   `json:"created_at"`
}

// ActivityInput is used when logging a user activity
type ActivityInput struct {
	NewsID       interface{} `json:"news_id" binding:"required"`
	ActivityType string      `json:"activity_type" binding:"required"`
}

// UserNewsPreference represents a user's preference for a news category
type UserNewsPreference struct {
	ID        int64     `json:"id"`
	UserID    int64     `json:"user_id"`
	Category  string    `json:"category"`
	Weight    float64   `json:"weight"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// RecommendationInput is used when requesting recommendations
type RecommendationInput struct {
	Limit int `json:"limit,omitempty" binding:"max=100"` // Default to 10 in handler
}
