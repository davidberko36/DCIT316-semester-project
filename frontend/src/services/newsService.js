import api from './api';

// News service for handling all news-related API calls
const newsService = {
  // Get all news with optional filtering
  getNews: async (params = {}) => {
    const response = await api.get('/news', { params });
    return response.data;
  },

  // Get latest news (simplified for homepage)
  getLatestNews: async (limit = 10) => {
    const response = await api.get('/news', { 
      params: { limit } 
    });
    return response.data;
  },

  // Get a specific news article by ID
  getNewsById: async (id) => {
    const response = await api.get(`/news/${id}`);
    return response.data;
  },

  // Get Ghana news
  getGhanaNews: async (category = '', limit = 10) => {
    const params = { limit };
    if (category) params.category = category;
    
    const response = await api.get('/external/ghana', { params });
    return response.data;
  },

  // Check fake news
  checkFakeNews: async (title, content) => {
    const response = await api.post('/news/detect', { title, content });
    return response.data;
  },

  // Get personalized recommendations (requires auth)
  getRecommendations: async (limit = 10) => {
    const response = await api.get('/user/recommendations', { params: { limit } });
    return response.data;
  },

  // Get filtered recommendations with fake news threshold (requires auth)
  getFilteredRecommendations: async (limit = 10, threshold = 0.3) => {
    const response = await api.get('/user/recommendations/filtered', { 
      params: { limit, threshold } 
    });
    return response.data;
  },

  // Log user activity with a news article (requires auth)
  logActivity: async (newsId, activityType) => {
    const response = await api.post('/news/activity', {
      news_id: newsId,
      activity_type: activityType
    });
    return response.data;
  },

  // Fetch external news
  fetchExternalNews: async () => {
    const response = await api.get('/news/fetch');
    return response.data;
  }
};

export default newsService;