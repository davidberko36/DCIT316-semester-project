package handlers

import (
	"github.com/davidberko36/DCIT316-semester-project/backend/internal/middleware"
	"github.com/gin-gonic/gin"
)

// Router sets up all API routes
func SetupRouter(
	userHandler *UserHandler,
	newsHandler *NewsHandler,
	authMiddleware *middleware.AuthMiddleware,
) *gin.Engine {
	router := gin.Default()

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

		// News routes - some public, some protected
		news := api.Group("/news")
		{
			news.GET("/", newsHandler.GetNews)               // Public endpoint to get news
			news.GET("/:id", newsHandler.GetNewsById)        // Public endpoint to get a specific news article
			news.POST("/detect", newsHandler.DetectFakeNews) // Public endpoint to detect fake news

			// Protected news routes
			protectedNews := news.Group("/")
			protectedNews.Use(authMiddleware.RequireAuth())
			{
				protectedNews.POST("/", newsHandler.CreateNews)
				protectedNews.PUT("/:id", newsHandler.UpdateNews)
				protectedNews.DELETE("/:id", newsHandler.DeleteNews)
				protectedNews.POST("/activity", newsHandler.LogActivity)
			}
		}

		// User routes - all protected
		user := api.Group("/user")
		user.Use(authMiddleware.RequireAuth())
		{
			user.GET("/profile", userHandler.GetProfile)
			user.PUT("/profile", userHandler.UpdateProfile)
			user.GET("/recommendations", newsHandler.GetRecommendations)
			user.GET("/activity", newsHandler.GetUserActivity)
		}
	}

	return router
}
