# Integrated ML System Documentation

This document outlines how the integrated ML system works, combining fake news detection and news recommendation in a unified workflow.

## System Components

1. **Fake News Detector**: Uses logistic regression to classify news articles
   - Implemented in `scripts/logistic_regression.py`
   - Exported as ONNX model to `models/lr_pipeline.onnx`
   - Integrated via `backend/internal/services/ml/fake_news.go`

2. **News Recommender**: Uses neural networks to generate user-specific news recommendations
   - Implemented in `scripts/neural_nets.py`
   - Exported as ONNX model to `models/recommender_scoring.onnx`
   - Integrated via `backend/internal/services/ml/recommendation.go`

3. **Composite News Fetcher**: Fetches news from multiple sources with fallback
   - Supports NewsAPI and APITube with country-specific queries
   - Uses fallback mechanisms when primary source fails
   - Implementation in `backend/internal/services/fetcher/composite_fetcher.go`

## Integrated Workflow

The system integrates fake news detection and recommendation in the following workflow:

1. **News Fetching**:
   - News articles are fetched from external APIs using the composite fetcher
   - If top headlines aren't available, fallback to keyword searches
   - Supports Ghana-specific news via the APITube integration

2. **Fake News Classification**:
   - Each fetched article is processed through the fake news detection model
   - Articles receive a fake news probability score (0-1)
   - High-probability fake news articles are flagged with `is_fake=true`

3. **News Storage**:
   - All articles are stored in the database with their fake news probability
   - Duplicates are automatically detected and skipped

4. **Recommendation Generation**:
   - User requests recommendations via API endpoints
   - System retrieves articles suitable for recommendation (newer, relevant)
   - User profile and activity history affect recommendations
   - Standard recommendations exclude articles flagged as fake (`is_fake=true`)

5. **Enhanced Filtering** (New Feature):
   - Filtered recommendations apply stricter threshold to fake news probability
   - Uses neural network model for recommendation scoring
   - Available via `/api/user/recommendations/filtered` endpoint
   - Customizable threshold for fake news filtering

## API Endpoints

- `/api/news/detect`: Detect fake news in a provided article
- `/api/user/recommendations`: Get personalized news recommendations
- `/api/user/recommendations/filtered`: Get recommendations with enhanced fake news filtering
- `/api/external/ghana`: Get Ghana-specific news from external sources

## Model Integration Details

### Fake News Detection

The system uses the ONNX-exported logistic regression model to classify news articles:

```go
// Example from news_service.go
isFake, probability, err := s.fakeNewsDetector.IsFake(input.Title, input.Content, threshold)
```

### News Recommendation

The recommendation engine uses a neural network to score articles based on user preferences:

```go
// Example from news_service.go
recommendations, err := s.recommendationEngine.GetRecommendations(user, activities, availableNews, limit)
```

### Enhanced Filtering

The new filtered recommendations feature applies stricter fake news filtering:

```go
// Example from news_service_filtered.go
func (s *NewsServiceImpl) GetFilteredRecommendations(userID int64, limit int, fakeNewsProbabilityThreshold float64) ([]models.News, error) {
    // Get standard recommendations first
    recommendations, err := s.GetRecommendations(userID, limit*2)
    if err != nil {
        return nil, fmt.Errorf("error getting recommendations: %w", err)
    }

    // Apply stricter fake news filtering
    var filteredRecommendations []models.News
    for _, news := range recommendations {
        if news.FakeProbability < fakeNewsProbabilityThreshold {
            filteredRecommendations = append(filteredRecommendations, news)
            if len(filteredRecommendations) >= limit {
                break
            }
        }
    }

    return filteredRecommendations, nil
}
```

## Configuration

The system supports configuration via environment variables:

- `FAKE_NEWS_THRESHOLD`: Default threshold for fake news classification (default: 0.5)
- `NEWS_API_KEY`: API key for NewsAPI
- `APITUBE_API_KEY`: API key for APITube
- `PREFER_NEWSAPI`: Set to any value to prefer NewsAPI over APITube (default prefers APITube)

## Testing

The integrated system includes test cases that verify both ML models work correctly together:

- `integrated_ml_test.go`: Tests the integrated workflow with mock implementations
- `mock_ml.go`: Mock implementations of ML interfaces for testing
- `mock_news_service.go`: Mock implementation of the news service interface

## Future Improvements

1. Enhance fake news detection model with more training data
2. Implement feedback mechanism to improve ML models over time
3. Add more news sources to the composite fetcher
4. Add explainability features to help users understand why articles are flagged as fake
5. Develop a front-end interface to visualize the ML results