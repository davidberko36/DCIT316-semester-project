package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/handlers"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/services/ml"
	"github.com/joho/godotenv"
	_ "github.com/mattn/go-sqlite3"
)

func main() {
	// Load .env file if it exists
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: No .env file found")
	}

	// Get port from environment or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize database
	db, err := initDatabase()
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// Initialize ML models
	fakeNewsDetector, recommendationEngine := initializeMLModels()

	// Initialize middleware
	authMiddleware := middleware.NewAuthMiddleware()

	// Initialize services
	userService := services.NewUserService(db, authMiddleware)
	newsService := services.NewNewsService(db, fakeNewsDetector, recommendationEngine)

	// Initialize and start the news scheduler
	newsScheduler := services.NewNewsScheduler(newsService)
	newsScheduler.Start()
	defer newsScheduler.Stop()

	// Initialize handlers
	userHandler := handlers.NewUserHandler(userService)
	newsHandler := handlers.NewNewsHandler(newsService)
	externalHandler := handlers.NewNewsExternalHandler(newsService)

	// Set up router with handlers
	router := handlers.SetupRouter(userHandler, newsHandler, authMiddleware, externalHandler)

	// Run the server
	log.Printf("Server starting on port %s\n", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func initializeMLModels() (*ml.FakeNewsDetector, *ml.RecommendationEngine) {
	// Get model paths
	modelsDir := os.Getenv("MODELS_DIR")
	if modelsDir == "" {
		// Default to models directory at the project root
		modelsDir = "../models"
	}

	// Initialize fake news detector
	fakeNewsModelPath := filepath.Join(modelsDir, "lr_pipeline.onnx")
	fakeNewsDetector, err := ml.NewFakeNewsDetector(fakeNewsModelPath)
	if err != nil {
		log.Printf("Warning: Could not load fake news model: %v", err)
		log.Println("Using simplified fake news detection")
		// Use a dummy model path that will just do basic text analysis
		fakeNewsDetector = &ml.FakeNewsDetector{}
	}

	// Initialize recommendation engine
	recommenderModelPath := filepath.Join(modelsDir, "recommender_scoring.onnx")
	recommender, err := ml.NewRecommendationEngine(recommenderModelPath)
	if err != nil {
		log.Printf("Warning: Could not load recommender model: %v", err)
		log.Println("Using simplified recommendation system")
		// Use a dummy model path that will just do basic recommendations
		recommender = &ml.RecommendationEngine{}
	}

	return fakeNewsDetector, recommender
}

func initDatabase() (*sql.DB, error) {
	dbPath := os.Getenv("DB_PATH")
	if dbPath == "" {
		dbPath = "./backend.db"
	}

	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory for database: %w", err)
		}
	}

	// Open database
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Initialize schema
	if err := initSchema(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return db, nil
}

func initSchema(db *sql.DB) error {
	// Create users table
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT NOT NULL UNIQUE,
			email TEXT NOT NULL UNIQUE,
			password_hash TEXT NOT NULL,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		)
	`)
	if err != nil {
		return err
	}

	// Create user_profiles table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS user_profiles (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			age INTEGER,
			gender TEXT,
			location TEXT,
			interests TEXT,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL,
			FOREIGN KEY (user_id) REFERENCES users (id)
		)
	`)
	if err != nil {
		return err
	}

	// Create news table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS news (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			title TEXT NOT NULL,
			content TEXT NOT NULL,
			url TEXT,
			source TEXT,
			category TEXT,
			is_fake BOOLEAN NOT NULL,
			fake_probability REAL NOT NULL,
			created_at DATETIME NOT NULL
		)
	`)
	if err != nil {
		return err
	}

	// Create user_activity table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS user_activity (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			news_id INTEGER NOT NULL,
			activity_type TEXT NOT NULL,
			created_at DATETIME NOT NULL,
			FOREIGN KEY (user_id) REFERENCES users (id),
			FOREIGN KEY (news_id) REFERENCES news (id)
		)
	`)
	if err != nil {
		return err
	}

	return nil
}

// This function is no longer needed since we're using handlers.SetupRouter
/*
func setupRoutes(r *gin.Engine, fakeNewsDetector *ml.FakeNewsDetector, recommender *ml.RecommendationEngine) {
	// Implementation removed for clarity
}
*/
