import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

// Protected route component that redirects to login if not authenticated
const ProtectedRoute = () => {
  const { isLoggedIn, loading } = useAuth();
  
  // Show loading state if still checking authentication
  if (loading) {
    return <div>Loading...</div>;
  }
  
  // Redirect to login if not authenticated
  return isLoggedIn ? <Outlet /> : <Navigate to="/login" />;
};

export default ProtectedRoute;