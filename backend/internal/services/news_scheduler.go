package services

import (
	"log"
	"sync"
	"time"
)

// NewsScheduler handles periodic news fetching and cleanup
type NewsScheduler struct {
	newsService     NewsService
	fetchInterval   time.Duration
	cleanupAge      time.Duration
	stopChan        chan struct{}
	wg              sync.WaitGroup
	isRunning       bool
	fetchCategories []string
	country         string
	articlesPerCat  int
}

// NewNewsScheduler creates a new scheduler for news operations
func NewNewsScheduler(newsService NewsService) *NewsScheduler {
	return &NewsScheduler{
		newsService:     newsService,
		fetchInterval:   10 * time.Minute,
		cleanupAge:      72 * time.Hour,
		stopChan:        make(chan struct{}),
		fetchCategories: []string{"general", "business", "technology", "sports", "entertainment", "health", "science"},
		country:         "gh", // Default to Ghana
		articlesPerCat:  10,   // Default 10 articles per category
	}
}

// Start begins the periodic news fetching and cleanup
func (s *NewsScheduler) Start() {
	if s.isRunning {
		return
	}

	s.isRunning = true
	s.wg.Add(2) // One for fetching, one for cleanup

	// Initial fetch
	go func() {
		log.Println("Performing initial news fetch...")
		s.fetchNews()
	}()

	// Periodic fetch
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(s.fetchInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				log.Println("Scheduled news fetch triggered")
				s.fetchNews()
			case <-s.stopChan:
				log.Println("Stopping news fetch scheduler")
				return
			}
		}
	}()

	// Periodic cleanup
	go func() {
		defer s.wg.Done()
		ticker := time.NewTicker(s.fetchInterval * 3) // Less frequent cleanup
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				log.Println("Scheduled news cleanup triggered")
				s.cleanupOldNews()
			case <-s.stopChan:
				log.Println("Stopping news cleanup scheduler")
				return
			}
		}
	}()

	log.Printf("News scheduler started with fetch interval: %v, cleanup age: %v", s.fetchInterval, s.cleanupAge)
}

// Stop halts the news scheduler
func (s *NewsScheduler) Stop() {
	if !s.isRunning {
		return
	}

	close(s.stopChan)
	s.wg.Wait()
	s.isRunning = false
	log.Println("News scheduler stopped")
}

// fetchNews fetches news for each category
func (s *NewsScheduler) fetchNews() {
	totalFetched := 0

	for _, category := range s.fetchCategories {
		log.Printf("Fetching %s news...", category)

		news, err := s.newsService.FetchAndStoreNews(s.country, category, s.articlesPerCat)
		if err != nil {
			log.Printf("Error fetching %s news: %v", category, err)
			continue
		}

		totalFetched += len(news)
		log.Printf("Successfully fetched %d %s news articles", len(news), category)
	}

	log.Printf("Fetch complete. Total articles added: %d", totalFetched)
}

// cleanupOldNews removes news older than the cleanup age
func (s *NewsScheduler) cleanupOldNews() {
	count, err := s.newsService.DeleteOldNews(s.cleanupAge)
	if err != nil {
		log.Printf("Error cleaning up old news: %v", err)
		return
	}

	log.Printf("Cleanup complete. %d old articles removed", count)
}
