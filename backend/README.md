# News Recommender Backend

This is the backend for the Ghana News Recommender System, implementing:
- Fake news detection using logistic regression
- News recommendation using neural networks

## Setup

1. **Environment Setup**:
   ```bash
   # Ensure you're in the backend directory
   cd backend

   # Install dependencies
   go mod download
   ```

2. **Environment Variables**:
   Create a `.env` file with the following variables:
   ```
   PORT=8080
   DB_PATH=./backend.db
   MODELS_DIR=../models
   JWT_SECRET=your-secret-key-should-be-changed-in-production
   ```

3. **Run the Server**:
   ```bash
   go run cmd/main.go
   ```

## API Endpoints

### Authentication
- `POST /api/auth/signup` - Register a new user
- `POST /api/auth/login` - Login and get JWT token

### News
- `GET /api/news/` - Get news articles (optionally filtered by category)
- `GET /api/news/:id` - Get a specific news article
- `POST /api/news/detect` - Detect if a news article is fake
- `POST /api/news/` - Create a new news article (requires authentication)
- `PUT /api/news/:id` - Update a news article (requires authentication)
- `DELETE /api/news/:id` - Delete a news article (requires authentication)
- `POST /api/news/activity` - Log user activity (requires authentication)

### User
- `GET /api/user/profile` - Get user profile (requires authentication)
- `PUT /api/user/profile` - Update user profile (requires authentication)
- `GET /api/user/recommendations` - Get personalized news recommendations (requires authentication)
- `GET /api/user/activity` - Get user activity history (requires authentication)

### Health Check
- `GET /health` - Check if the API is running

## ML Models

The backend uses two ML models exported in ONNX format:
- `lr_pipeline.onnx` - Logistic regression model for fake news detection
- `recommender_scoring.onnx` - Neural network model for news recommendation

If the models aren't available, the system falls back to simplified implementations for testing purposes.

## Development

This backend uses:
- **Gin Web Framework**: For routing and API handling
- **SQLite**: For persistent data storage
- **JWT**: For authentication and authorization
- **ONNX Runtime**: For machine learning model inference

## Project Structure

```
backend/
├── cmd/
│   └── main.go             # Application entry point
├── internal/
│   ├── handlers/           # HTTP request handlers
│   ├── middleware/         # HTTP middleware (authentication, etc.)
│   ├── models/             # Data models
│   └── services/           # Business logic
│       └── ml/             # Machine learning services
├── .env                    # Environment variables
└── go.mod                  # Go module definition
```