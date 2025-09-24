// Package middleware provides HTTP middleware components for the application
package middleware

import (
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// NewCORSMiddleware returns a configured CORS middleware for handling cross-origin requests
func NewCORSMiddleware() gin.HandlerFunc {
	return cors.New(cors.Config{
		// AllowOrigins defines the origins that are allowed to make requests
		AllowOrigins: []string{
			"http://localhost:3000",        // Development frontend
			"http://localhost:5000",        // Production frontend
			"https://news-app.example.com", // Production domain (replace with actual domain)
		},
		// AllowMethods defines the methods that are allowed for CORS requests
		AllowMethods: []string{
			"GET",
			"POST",
			"PUT",
			"PATCH",
			"DELETE",
			"OPTIONS",
		},
		// AllowHeaders defines the headers that are allowed in requests
		AllowHeaders: []string{
			"Origin",
			"Content-Type",
			"Content-Length",
			"Accept-Encoding",
			"X-CSRF-Token",
			"Authorization",
			"Accept",
			"X-Requested-With",
		},
		// ExposeHeaders defines the headers that can be read by the client
		ExposeHeaders: []string{
			"Content-Length",
			"Content-Type",
		},
		// AllowCredentials allows cookies to be sent with requests
		AllowCredentials: true,
		// MaxAge defines how long (in seconds) the results of a preflight request can be cached
		MaxAge: 12 * time.Hour,
	})
}
