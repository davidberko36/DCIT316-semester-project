// Add CORS middleware to the Go backend in routes.go

package handlers

import (
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Router sets up all API routes
func SetupRouter(
	userHandler *UserHandler,
	newsHandler *NewsHandler,
	authMiddleware *middleware.AuthMiddleware,
	externalHandler *NewsExternalHandler,
) *gin.Engine {
	router := gin.Default()

	// Configure CORS middleware
	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"}, // Update with your frontend URL
		AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           86400, // 24 hours
	}))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":  "ok",
			"message": "News recommender API is running",
		})
	})

	// API group
	api := router.Group("/api")
	{
		// Public routes
		auth := api.Group("/auth")
		{
			auth.POST("/signup", userHandler.Signup)
			auth.POST("/login", userHandler.Login)
		}

		// Rest of the routes...