package services

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/models"
	"golang.org/x/crypto/bcrypt"
)

// UserServiceImpl implements the UserService interface
type UserServiceImpl struct {
	db             *sql.DB
	authMiddleware *middleware.AuthMiddleware
}

// NewUserService creates a new user service
func NewUserService(db *sql.DB, authMiddleware *middleware.AuthMiddleware) *UserServiceImpl {
	return &UserServiceImpl{
		db:             db,
		authMiddleware: authMiddleware,
	}
}

// CreateUser creates a new user account
func (s *UserServiceImpl) CreateUser(username, email, password string) (*models.User, string, error) {
	// Check if username or email already exists
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM users WHERE username = ? OR email = ?", username, email).Scan(&count)
	if err != nil {
		return nil, "", err
	}

	if count > 0 {
		return nil, "", fmt.Errorf("username or email already exists")
	}

	// Hash the password
	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, "", err
	}

	// Insert the new user
	now := time.Now()
	query := `
		INSERT INTO users (username, email, password_hash, created_at, updated_at) 
		VALUES (?, ?, ?, ?, ?)
	`
	result, err := s.db.Exec(query, username, email, string(hashedPassword), now, now)
	if err != nil {
		return nil, "", err
	}

	// Get the user ID
	id, err := result.LastInsertId()
	if err != nil {
		return nil, "", err
	}

	// Create an empty profile
	profileQuery := `
		INSERT INTO user_profiles (user_id, created_at, updated_at) 
		VALUES (?, ?, ?)
	`
	_, err = s.db.Exec(profileQuery, id, now, now)
	if err != nil {
		// Not critical, continue even if profile creation fails
		fmt.Printf("Warning: Failed to create profile for user %d: %v\n", id, err)
	}

	// Generate a JWT token
	token, err := s.authMiddleware.GenerateToken(id)
	if err != nil {
		return nil, "", err
	}

	// Return the user
	user := &models.User{
		ID:           id,
		Username:     username,
		Email:        email,
		PasswordHash: string(hashedPassword),
		CreatedAt:    now,
		UpdatedAt:    now,
		Profile:      &models.UserProfile{UserID: id, CreatedAt: now, UpdatedAt: now},
	}

	return user, token, nil
}

// AuthenticateUser authenticates a user and returns a JWT token
func (s *UserServiceImpl) AuthenticateUser(username, password string) (*models.User, string, error) {
	// Find the user
	var user models.User
	query := `
		SELECT id, username, email, password_hash, created_at, updated_at 
		FROM users 
		WHERE username = ?
	`
	err := s.db.QueryRow(query, username).Scan(
		&user.ID,
		&user.Username,
		&user.Email,
		&user.PasswordHash,
		&user.CreatedAt,
		&user.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, "", fmt.Errorf("invalid username or password")
		}
		return nil, "", err
	}

	// Check the password
	err = bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(password))
	if err != nil {
		return nil, "", fmt.Errorf("invalid username or password")
	}

	// Generate a JWT token
	token, err := s.authMiddleware.GenerateToken(user.ID)
	if err != nil {
		return nil, "", err
	}

	// Get the user profile
	profile, err := s.getUserProfile(user.ID)
	if err == nil {
		user.Profile = profile
	}

	return &user, token, nil
}

// GetUserByID retrieves a user by ID
func (s *UserServiceImpl) GetUserByID(id int64) (*models.User, error) {
	// Find the user
	var user models.User
	query := `
		SELECT id, username, email, password_hash, created_at, updated_at 
		FROM users 
		WHERE id = ?
	`
	err := s.db.QueryRow(query, id).Scan(
		&user.ID,
		&user.Username,
		&user.Email,
		&user.PasswordHash,
		&user.CreatedAt,
		&user.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("user not found")
		}
		return nil, err
	}

	// Get the user profile
	profile, err := s.getUserProfile(id)
	if err == nil {
		user.Profile = profile
	}

	return &user, nil
}

// UpdateUserProfile updates a user's profile
func (s *UserServiceImpl) UpdateUserProfile(id int64, update models.ProfileUpdate) (*models.User, error) {
	// Check if user exists
	user, err := s.GetUserByID(id)
	if err != nil {
		return nil, err
	}

	// Check if profile exists
	profile, err := s.getUserProfile(id)
	if err != nil {
		// Create profile if it doesn't exist
		now := time.Now()
		query := `
			INSERT INTO user_profiles (user_id, created_at, updated_at) 
			VALUES (?, ?, ?)
		`
		_, err = s.db.Exec(query, id, now, now)
		if err != nil {
			return nil, err
		}

		profile = &models.UserProfile{
			UserID:    id,
			CreatedAt: now,
			UpdatedAt: now,
		}
	}

	// Update profile
	query := `
		UPDATE user_profiles 
		SET age = ?, gender = ?, location = ?, interests = ?, updated_at = ? 
		WHERE user_id = ?
	`
	now := time.Now()
	_, err = s.db.Exec(
		query,
		update.Age,
		update.Gender,
		update.Location,
		update.Interests,
		now,
		id,
	)
	if err != nil {
		return nil, err
	}

	// Update user's profile
	profile.Age = update.Age
	profile.Gender = update.Gender
	profile.Location = update.Location
	profile.Interests = update.Interests
	profile.UpdatedAt = now
	user.Profile = profile

	return user, nil
}

// getUserProfile retrieves a user's profile
func (s *UserServiceImpl) getUserProfile(userID int64) (*models.UserProfile, error) {
	query := `
		SELECT id, user_id, age, gender, location, interests, created_at, updated_at 
		FROM user_profiles 
		WHERE user_id = ?
	`

	var profile models.UserProfile
	var age sql.NullInt32
	var gender, location, interests sql.NullString

	err := s.db.QueryRow(query, userID).Scan(
		&profile.ID,
		&profile.UserID,
		&age,
		&gender,
		&location,
		&interests,
		&profile.CreatedAt,
		&profile.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	// Convert SQL nullable types to Go pointers
	if age.Valid {
		ageInt := int(age.Int32)
		profile.Age = &ageInt
	}
	if gender.Valid {
		profile.Gender = &gender.String
	}
	if location.Valid {
		profile.Location = &location.String
	}
	if interests.Valid {
		profile.Interests = &interests.String
	}

	return &profile, nil
}
