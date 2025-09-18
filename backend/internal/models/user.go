package models

import (
	"time"
)

// User represents a user in the system
type User struct {
	ID           int64        `json:"id"`
	Username     string       `json:"username"`
	Email        string       `json:"email"`
	PasswordHash string       `json:"-"` // Don't expose password hash
	CreatedAt    time.Time    `json:"created_at"`
	UpdatedAt    time.Time    `json:"updated_at"`
	Profile      *UserProfile `json:"profile,omitempty"`
}

// UserProfile contains additional user information for recommendations
type UserProfile struct {
	ID        int64     `json:"id"`
	UserID    int64     `json:"user_id"`
	Age       *int      `json:"age,omitempty"`
	Gender    *string   `json:"gender,omitempty"`
	Location  *string   `json:"location,omitempty"`
	Interests *string   `json:"interests,omitempty"` // Comma-separated interests
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UserSignup is used for the signup request
type UserSignup struct {
	Username string `json:"username" binding:"required"`
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required,min=6"`
}

// UserLogin is used for the login request
type UserLogin struct {
	Username string `json:"username" binding:"required"`
	Password string `json:"password" binding:"required"`
}

// ProfileUpdate is used to update a user's profile
type ProfileUpdate struct {
	Age       *int    `json:"age,omitempty"`
	Gender    *string `json:"gender,omitempty"`
	Location  *string `json:"location,omitempty"`
	Interests *string `json:"interests,omitempty"`
}
