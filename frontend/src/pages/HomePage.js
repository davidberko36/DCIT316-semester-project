import React, { useState, useEffect } from 'react';
import { Row, Col, Alert, Spinner, Nav, Tab, Card } from 'react-bootstrap';
import Layout from '../components/Layout';
import NewsCard from '../components/NewsCard';
import newsService from '../services/newsService';

const HomePage = () => {
  const [news, setNews] = useState([]);
  const [serpApiNews, setSerpApiNews] = useState([]);
  const [activeSource, setActiveSource] = useState('default');
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
        
        // Fetch SerpAPI news as well
        try {
          const serpData = await newsService.getSerpApiNews();
          if (serpData.articles && Array.isArray(serpData.articles)) {
            setSerpApiNews(serpData.articles);
          }
        } catch (serpErr) {
          console.error('Error fetching SerpAPI news:', serpErr);
          // We don't set the main error state here to allow regular news to display
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

  // Get the active news articles based on selected source
  const getActiveNews = () => {
    switch (activeSource) {
      case 'serpapi':
        return serpApiNews;
      case 'default':
      default:
        return news;
    }
  };

  const activeNews = getActiveNews();

  return (
    <Layout>
      <h1 className="mb-4">Latest Ghana News</h1>
      
      <Card className="mb-4">
        <Card.Header>
          <Nav variant="tabs" defaultActiveKey="default" onSelect={(key) => setActiveSource(key)}>
            <Nav.Item>
              <Nav.Link eventKey="default">Standard Source</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="serpapi">SerpAPI Source</Nav.Link>
            </Nav.Item>
          </Nav>
        </Card.Header>
        <Card.Body>
          <p className="text-muted">
            {activeSource === 'serpapi' 
              ? 'Displaying news from SerpAPI - a comprehensive news aggregation service.'
              : 'Displaying news from our standard sources.'}
          </p>
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
          {activeNews.map((article, index) => (
            <Col key={article.id || `${article.url}-${index}`}>
              <NewsCard article={article} />
            </Col>
          ))}
          
          {activeNews.length === 0 && !loading && !error && (
            <Col xs={12}>
              <Alert variant="info">
                No news articles available from {activeSource === 'serpapi' ? 'SerpAPI' : 'standard sources'} at the moment.
              </Alert>
            </Col>
          )}
        </Row>
      )}
    </Layout>
  );
};

export default HomePage;