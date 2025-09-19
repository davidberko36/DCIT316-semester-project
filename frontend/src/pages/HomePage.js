import React, { useState, useEffect } from 'react';
import { Row, Col, Alert, Spinner } from 'react-bootstrap';
import Layout from '../components/Layout';
import NewsCard from '../components/NewsCard';
import newsService from '../services/newsService';

const HomePage = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        setLoading(true);
        const data = await newsService.getLatestNews();
        
        // Handle different possible response structures
        if (Array.isArray(data)) {
          setNews(data);
        } else if (data.articles && Array.isArray(data.articles)) {
          setNews(data.articles);
        } else {
          setNews([]);
          console.warn('Unexpected response format:', data);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching news:', err);
        setError('Failed to load the latest news. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchNews();
  }, []);

  return (
    <Layout>
      <h1 className="mb-4">Latest Ghana News</h1>
      
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
          {news.map((article, index) => (
            <Col key={article.id || `${article.url}-${index}`}>
              <NewsCard article={article} />
            </Col>
          ))}
          
          {news.length === 0 && !loading && !error && (
            <Col xs={12}>
              <Alert variant="info">No news articles available at the moment.</Alert>
            </Col>
          )}
        </Row>
      )}
    </Layout>
  );
};

export default HomePage;