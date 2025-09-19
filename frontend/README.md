# Ghana News Portal Frontend

This is the frontend implementation for the Ghana News Portal, a semester project for DCIT316. The application integrates with a Go backend to provide news articles from Ghana using both NewsAPI and APITube sources, with fake news detection and personalized recommendations.

## Features

- Browse latest news from Ghana
- Search for specific news topics
- View detailed news articles
- User authentication (login/register)
- Fake news detection using machine learning
- Personalized news recommendations

## Architecture

This frontend is built with:

- React.js - Frontend library
- React Router - Navigation
- React Bootstrap - UI components
- Axios - API communication

## Project Structure

```
frontend/
  ├── public/
  ├── src/
  │   ├── components/        # Reusable components
  │   ├── context/           # Context providers
  │   ├── hooks/             # Custom React hooks
  │   ├── pages/             # Page components
  │   ├── routes/            # Routing components
  │   ├── services/          # API services
  │   ├── App.js             # Main component
  │   ├── App.css            # Global styles
  │   └── index.js           # Entry point
  └── package.json           # Dependencies
```

## Setup and Running

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm start
   ```

3. Build for production:
   ```
   npm run build
   ```

## Backend Integration

The frontend integrates with the backend through the service layer:

- `api.js` - Base API configuration with authentication handling
- `authService.js` - Authentication operations (login, register, etc.)
- `newsService.js` - News operations (fetching, searching, recommendations)

## Authentication

The application uses JWT-based authentication with token storage in localStorage. The AuthContext provides authentication state throughout the application.

## Protected Routes

Some features require authentication:

- Personalized recommendations
- Fake news detection

These routes are protected using the ProtectedRoute component, which redirects unauthenticated users to the login page.

## Available Scripts (from Create React App)

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

### `npm test`

Launches the test runner in the interactive watch mode.

### `npm run build`

Builds the app for production to the `build` folder.
