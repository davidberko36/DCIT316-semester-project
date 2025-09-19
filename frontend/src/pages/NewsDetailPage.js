import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Badge, Button, Alert, Spinner } from 'react-bootstrap';
import Layout from '../components/Layout';
import newsService from '../services/newsService';
import { useAuth } from '../hooks/useAuth';

const NewsDetailPage = () => {
  const { id } = useParams();
  const [article, setArticle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [fakeNewsResult, setFakeNewsResult] = useState(null);
  const [checkingFakeNews, setCheckingFakeNews] = useState(false);
  const { isLoggedIn } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const fetchArticle = async () => {
      try {
        setLoading(true);
        const data = await newsService.getNewsById(id);
        setArticle(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching article:', err);
        setError('Failed to load the article. It may have been removed or is unavailable.');
      } finally {
        setLoading(false);
      }
    };

    fetchArticle();
  }, [id]);

  const handleCheckFakeNews = async () => {
    if (!isLoggedIn) {
      navigate('/login', { state: { from: `/news/${id}` } });
      return;
    }

    try {
      setCheckingFakeNews(true);
      const result = await newsService.checkFakeNews(article.title, article.content || article.description);
      setFakeNewsResult(result);
    } catch (err) {
      console.error('Error checking fake news:', err);
      setError('Failed to check if this is fake news. Please try again later.');
    } finally {
      setCheckingFakeNews(false);
    }
  };

  // Format date to a readable format
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <Layout>
      {loading ? (
        <div className="text-center my-5">
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        </div>
      ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : article ? (
        <Container>
          <Row className="mb-4">
            <Col>
              <div className="d-flex justify-content-between align-items-center mb-3">
                <h1>{article.title}</h1>
                <Button 
                  variant="outline-primary" 
                  onClick={() => navigate(-1)}
                >
                  Back to News
                </Button>
              </div>
              
              <div className="d-flex justify-content-between align-items-center mb-4">
                <div>
                  <Badge bg="primary" className="me-2">{article.source?.name || 'Unknown Source'}</Badge>
                  <small className="text-muted">{formatDate(article.publishedAt)}</small>
                </div>
                
                {!fakeNewsResult && (
                  <Button 
                    variant="warning" 
                    onClick={handleCheckFakeNews}
                    disabled={checkingFakeNews}
                  >
                    {checkingFakeNews ? (
                      <>
                        <Spinner
                          as="span"
                          animation="border"
                          size="sm"
                          role="status"
                          aria-hidden="true"
                        />
                        <span className="ms-2">Checking...</span>
                      </>
                    ) : (
                      'Check Fake News'
                    )}
                  </Button>
                )}
              </div>
              
              {fakeNewsResult && (
                <Alert 
                  variant={fakeNewsResult.probability > 0.5 ? 'danger' : 'success'} 
                  className="mb-4"
                >
                  <Alert.Heading>
                    {fakeNewsResult.probability > 0.5 
                      ? 'This article may contain misinformation' 
                      : 'This article appears to be reliable'}
                  </Alert.Heading>
                  <p>
                    Our fake news detection system rates this article as {' '}
                    <strong>
                      {fakeNewsResult.probability > 0.5 
                        ? `${Math.round(fakeNewsResult.probability * 100)}% likely to be fake news` 
                        : `${Math.round((1 - fakeNewsResult.probability) * 100)}% likely to be real news`}
                    </strong>
                  </p>
                </Alert>
              )}
            </Col>
          </Row>
          
          {article.urlToImage && (
            <Row className="mb-4">
              <Col>
                <img 
                  src={article.urlToImage} 
                  alt={article.title}
                  className="img-fluid rounded shadow"
                  style={{ maxHeight: '500px', width: '100%', objectFit: 'cover' }}
                  onError={(e) => {
                    e.target.src = 'https://via.placeholder.com/800x400?text=No+Image+Available';
                  }}
                />
              </Col>
            </Row>
          )}
          
          <Row>
            <Col>
              <Card className="shadow-sm">
                <Card.Body>
                  {article.description && (
                    <Card.Subtitle className="mb-3 text-muted lead">
                      {article.description}
                    </Card.Subtitle>
                  )}
                  
                  {article.content ? (
                    <div dangerouslySetInnerHTML={{ __html: article.content }} />
                  ) : (
                    <p>No content available for this article.</p>
                  )}
                  
                  {article.url && (
                    <div className="mt-4">
                      <a 
                        href={article.url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="btn btn-outline-secondary"
                      >
                        Read full article at source
                      </a>
                    </div>
                  )}
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Container>
      ) : (
        <Alert variant="warning">No article found with the given ID.</Alert>
      )}
    </Layout>
  );
};

export default NewsDetailPage;