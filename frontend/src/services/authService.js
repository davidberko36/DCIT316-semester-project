import api from './api';

// Authentication service for handling user authentication
const authService = {
  // Register a new user
  register: async (username, email, password) => {
    const response = await api.post('/auth/signup', {
      username,
      email,
      password
    });
    return response.data;
  },

  // Login a user
  login: async (username, password) => {
    const response = await api.post('/auth/login', {
      username,
      password
    });
    
    // Store token and user data in localStorage
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
  },

  // Logout a user
  logout: () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  // Get current user
  getCurrentUser: () => {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      return JSON.parse(userStr);
    }
    return null;
  },

  // Check if user is logged in
  isLoggedIn: () => {
    return !!localStorage.getItem('token');
  },

  // Get user profile
  getProfile: async () => {
    const response = await api.get('/user/profile');
    return response.data;
  },

  // Update user profile
  updateProfile: async (profileData) => {
    const response = await api.put('/user/profile', profileData);
    return response.data;
  },

  // Get user activity
  getUserActivity: async (limit = 10, offset = 0) => {
    const response = await api.get('/user/activity', {
      params: { limit, offset }
    });
    return response.data;
  }
};

export default authService;