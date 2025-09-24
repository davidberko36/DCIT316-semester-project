// main.go - Entry point for the news API server
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/davidberko36/DCIT316-semester-project/backend/internal/handlers"
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables from .env file
	if err := godotenv.Load(); err != nil {
		log.Println("Warning: .env file not found, using environment variables")
	}

	// Get port from environment variable or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Initialize middleware
	authMiddleware := middleware.NewAuthMiddleware()

	// Initialize handlers
	userHandler := handlers.NewUserHandler()
	newsHandler := handlers.NewNewsHandler()
	externalHandler := handlers.NewNewsExternalHandler()

	// Set up router with middleware and handlers
	router := handlers.SetupRouter(
		userHandler,
		newsHandler,
		authMiddleware,
		externalHandler,
	)

	// Start the server
	fmt.Printf("Server running on port %s\n", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
