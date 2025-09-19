package services

import (
	"fmt"
	"time"
)

// DeleteOldNews removes news articles older than the specified duration
func (s *NewsServiceImpl) DeleteOldNews(age time.Duration) (int64, error) {
	// Calculate the cutoff time
	cutoffTime := time.Now().Add(-age)

	// Execute the delete query
	result, err := s.db.Exec(`DELETE FROM news WHERE created_at < ?`, cutoffTime)
	if err != nil {
		return 0, fmt.Errorf("error deleting old news: %w", err)
	}

	// Get the number of affected rows
	count, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("error getting affected rows: %w", err)
	}

	return count, nil
}
