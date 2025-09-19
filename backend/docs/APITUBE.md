# APITube Integration for Ghana News

This integration enables the application to fetch news specifically from Ghana using the APITube news API service.

## Features

- Fetches news articles specifically from Ghana
- Uses both APITube and NewsAPI with automatic fallback
- Supports filtering by category
- Extracts and analyzes content for fake news detection
- Categorizes articles automatically

## Environment Setup

To use this feature, add the following to your `.env` file:

```
# News API configuration
NEWS_API_KEY=your-newsapi-key-here
APITUBE_API_KEY=your-apitube-key-here
```

## Getting an APITube API Key

1. Sign up at [APITube.io](https://apitube.io/)
2. Navigate to your account dashboard
3. Create a new API key
4. Copy the key to your `.env` file

## API Endpoints

### Fetch Ghana News (without storing)

```
GET /api/external/ghana
```

Query parameters:
- `category`: Filter by category (general, business, technology, sports, health, entertainment, science)
- `limit`: Maximum number of articles to fetch (default: 10)

### Fetch and Store Ghana News

```
POST /api/external/ghana/store
```

Query parameters:
- `category`: Filter by category (general, business, technology, sports, health, entertainment, science)
- `limit`: Maximum number of articles to fetch (default: 10)

## Examples

### Fetch Business News from Ghana

```
GET /api/external/ghana?category=business&limit=5
```

### Fetch and Store Technology News

```
POST /api/external/ghana/store?category=technology&limit=10
```

## Implementation Details

This integration uses a composite fetcher pattern that tries multiple news sources with automatic fallback. If APITube doesn't return results, it will try NewsAPI, and vice versa. If both direct queries fail, it will try broader searches to ensure results are returned.

The system also includes automatic categorization using keyword analysis for articles that don't have explicit categories.