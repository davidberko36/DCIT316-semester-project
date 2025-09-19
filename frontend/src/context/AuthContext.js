import React, { createContext, useState, useEffect } from 'react';
import authService from '../services/authService';

// Create Authentication Context
export const AuthContext = createContext();

// Authentication Provider Component
export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load user from localStorage on initial render
  useEffect(() => {
    const user = authService.getCurrentUser();
    if (user) {
      setCurrentUser(user);
    }
    setLoading(false);
  }, []);

  // Login function
  const login = async (username, password) => {
    try {
      setLoading(true);
      setError(null);
      const data = await authService.login(username, password);
      setCurrentUser(data.user);
      return data;
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during login');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Register function
  const register = async (username, email, password) => {
    try {
      setLoading(true);
      setError(null);
      const data = await authService.register(username, email, password);
      return data;
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during registration');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  // Logout function
  const logout = () => {
    authService.logout();
    setCurrentUser(null);
  };

  // Value to be provided to consumers
  const value = {
    currentUser,
    loading,
    error,
    isLoggedIn: !!currentUser,
    login,
    register,
    logout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};