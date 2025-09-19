import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Form, Button, Alert, Spinner } from 'react-bootstrap';
import Layout from '../components/Layout';
import NewsCard from '../components/NewsCard';
import newsService from '../services/newsService';

const RecommendationsPage = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');

  useEffect(() => {
    // Fetch available categories
    const fetchCategories = async () => {
      try {
        // This should be replaced with actual API call when available
        const defaultCategories = ['general', 'business', 'technology', 'entertainment', 'sports', 'health', 'science'];
        setCategories(defaultCategories);
        if (defaultCategories.length > 0) {
          setSelectedCategory(defaultCategories[0]); // Set first category as default
        }
      } catch (err) {
        console.error('Error fetching categories:', err);
        setError('Failed to load news categories. Please try again later.');
      }
    };

    fetchCategories();
  }, []);

  useEffect(() => {
    // Fetch recommendations when category changes
    if (selectedCategory) {
      fetchRecommendations(selectedCategory);
    }
  }, [selectedCategory]);

  const fetchRecommendations = async (category) => {
    try {
      setLoading(true);
      const data = await newsService.getRecommendations(10);
      setRecommendations(data);
      setError(null);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError('Failed to load personalized recommendations. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleCategoryChange = (e) => {
    setSelectedCategory(e.target.value);
  };

  return (
    <Layout>
      <h1 className="mb-4">Personalized Recommendations</h1>
      
      <Card className="mb-4 shadow-sm">
        <Card.Body>
          <p className="lead">
            Our recommendation system analyzes your reading patterns to suggest articles that match your interests.
          </p>
          
          <Form.Group controlId="categorySelect" className="mb-0">
            <Form.Label>Filter by category:</Form.Label>
            <Form.Select
              value={selectedCategory}
              onChange={handleCategoryChange}
              disabled={loading || categories.length === 0}
            >
              {categories.map((category) => (
                <option key={category} value={category}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </option>
              ))}
            </Form.Select>
          </Form.Group>
        </Card.Body>
      </Card>
      
      {error && (
        <Alert variant="danger">{error}</Alert>
      )}
      
      {loading ? (
        <div className="text-center my-5">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      ) : (
        <Row xs={1} md={2} lg={3} className="g-4">
          {recommendations.map((article, index) => (
            <Col key={article.id || `${article.url}-${index}`}>
              <NewsCard 
                article={article} 
                showFakeNewsProbability={true} 
              />
            </Col>
          ))}
          
          {recommendations.length === 0 && !loading && !error && (
            <Col xs={12}>
              <Alert variant="info">
                No recommendations available for this category at the moment. Try selecting a different category or check back later.
              </Alert>
            </Col>
          )}
        </Row>
      )}
    </Layout>
  );
};

export default RecommendationsPage;