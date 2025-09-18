package ml

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// FakeNewsDetector provides methods for detecting fake news
type FakeNewsDetector struct {
	ModelPath string
}

// NewFakeNewsDetector loads a fake news detection model from an ONNX file
func NewFakeNewsDetector(modelPath string) (*FakeNewsDetector, error) {
	// Check if the model file exists
	absPath, err := filepath.Abs(modelPath)
	if err != nil {
		return nil, fmt.Errorf("error resolving model path: %w", err)
	}

	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found: %s", absPath)
	}

	return &FakeNewsDetector{
		ModelPath: absPath,
	}, nil
}

// Classify predicts whether a news article is fake or legitimate
// Returns probability of being fake (higher = more likely to be fake)
func (fd *FakeNewsDetector) Classify(title string, content string) (float64, error) {
	// This is a simplified implementation for testing
	// In a real implementation, we would use ONNX runtime to run the model

	// For testing, we'll use a simple heuristic
	text := strings.ToLower(title + " " + content)

	// Keywords that might indicate fake news
	fakeKeywords := []string{"fake", "hoax", "conspiracy", "clickbait", "false", "rumor"}
	count := 0

	for _, keyword := range fakeKeywords {
		if strings.Contains(text, keyword) {
			count++
		}
	}

	// Calculate a probability from 0 to 1
	probability := math.Min(1.0, float64(count)*0.2)

	return probability, nil
}

// IsFake determines if an article is fake based on a threshold
func (fd *FakeNewsDetector) IsFake(title string, content string, threshold float64) (bool, float64, error) {
	probability, err := fd.Classify(title, content)
	if err != nil {
		return false, 0, err
	}

	// Ensure probability is valid (handle NaN, Inf)
	if math.IsNaN(probability) || math.IsInf(probability, 0) {
		return false, 0, fmt.Errorf("invalid probability from model: %f", probability)
	}

	return probability >= threshold, probability, nil
}
