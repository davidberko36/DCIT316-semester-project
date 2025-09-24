import api from './api';

// News service for handling all news-related API calls
const newsService = {
  // Get all news with optional filtering
  getNews: async (params = {}) => {
    const response = await api.get('/news', { params });
    return response.data;
  },

  // Get latest news (simplified for homepage)
  getLatestNews: async (limit = 20) => {
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
  getGhanaNews: async (category = '', limit = 20) => {
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
  getRecommendations: async (limit = 20) => {
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
  },
  
  // Get news from multiple categories for Ghana
  getGhanaNewsByCategories: async (categories = ['general', 'business', 'technology', 'entertainment', 'sports', 'health', 'science'], limitPerCategory = 5) => {
    // This will be implemented on the backend
    // For now, we'll make separate requests and combine them
    const allNews = [];
    
    // Make parallel requests for all categories
    try {
      const requests = categories.map(category => 
        api.get('/external/ghana', { params: { category, limit: limitPerCategory } })
      );
      
      const responses = await Promise.all(requests);
      
      // Combine all articles
      responses.forEach(response => {
        if (response.data && response.data.articles) {
          allNews.push(...response.data.articles);
        }
      });
      
      return { articles: allNews };
    } catch (error) {
      console.error('Error fetching multi-category news:', error);
      throw error;
    }
  },

  // SerpAPI news sources
  getSerpApiNews: async (query = 'Ghana news', location = 'Ghana', limit = 20) => {
    try {
      const response = await api.get('/external/serpapi', {
        params: { query, location, limit }
      });
      
      // Transform the data to match our application's expected format
      if (response.data && response.data.news_results) {
        const articles = response.data.news_results.map(item => ({
          id: `serpapi-${item.position}`,
          title: item.title,
          description: item.snippet,
          content: item.snippet,
          url: item.link,
          source: item.source,
          publishedAt: item.published_at || item.date,
          image: item.thumbnail,
          category: 'news',
          fakeProbability: 0.5 // Default value, will be calculated by backend
        }));
        return { articles };
      }
      
      return { articles: [] };
    } catch (error) {
      console.error('Error fetching SerpAPI news:', error);
      throw error;
    }
  },
  
  // Get SerpAPI news by category
  getSerpApiNewsByCategory: async (category, location = 'Ghana', limit = 20) => {
    try {
      const query = `${category} news Ghana`;
      const response = await api.get('/external/serpapi', {
        params: { query, location, limit }
      });
      
      // Transform the data to match our application's expected format
      if (response.data && response.data.news_results) {
        const articles = response.data.news_results.map(item => ({
          id: `serpapi-${category}-${item.position}`,
          title: item.title,
          description: item.snippet,
          content: item.snippet,
          url: item.link,
          source: item.source,
          publishedAt: item.published_at || item.date,
          image: item.thumbnail,
          category: category,
          fakeProbability: 0.5 // Default value, will be calculated by backend
        }));
        return { articles };
      }
      
      return { articles: [] };
    } catch (error) {
      console.error(`Error fetching SerpAPI ${category} news:`, error);
      throw error;
    }
  }
};

export default newsService;