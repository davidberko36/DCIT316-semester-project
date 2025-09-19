import React, { useState } from 'react';
import { Form, Button, Card, Alert, Spinner } from 'react-bootstrap';
import Layout from '../components/Layout';
import newsService from '../services/newsService';

const FakeNewsCheckPage = () => {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validated, setValidated] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const form = e.currentTarget;
    if (form.checkValidity() === false) {
      e.stopPropagation();
      setValidated(true);
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      const checkResult = await newsService.checkFakeNews(title, content);
      setResult(checkResult);
    } catch (err) {
      console.error('Error checking fake news:', err);
      setError('Failed to analyze the text. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setTitle('');
    setContent('');
    setResult(null);
    setError(null);
    setValidated(false);
  };

  return (
    <Layout>
      <h1 className="mb-4">Fake News Detector</h1>
      
      <Card className="mb-4 shadow-sm">
        <Card.Body>
          <p className="lead">
            Our fake news detection system uses machine learning to analyze text and determine the likelihood
            that it contains misinformation. Enter a news headline and content to check its credibility.
          </p>
        </Card.Body>
      </Card>
      
      {error && (
        <Alert variant="danger">{error}</Alert>
      )}
      
      {result ? (
        <Card className="shadow">
          <Card.Header className={`bg-${result.probability > 0.5 ? 'danger' : 'success'} text-white`}>
            <h3 className="mb-0">Analysis Result</h3>
          </Card.Header>
          <Card.Body>
            <h4 className="mb-3">
              {result.probability > 0.5
                ? 'This content may contain misinformation'
                : 'This content appears to be reliable'}
            </h4>
            
            <div className="mb-3">
              <strong>Confidence Score:</strong>{' '}
              {result.probability > 0.5
                ? `${Math.round(result.probability * 100)}% likely to be fake news`
                : `${Math.round((1 - result.probability) * 100)}% likely to be real news`}
            </div>
            
            <div className="mb-4">
              <h5>Analyzed Content:</h5>
              <Card className="bg-light">
                <Card.Body>
                  <h6>{title}</h6>
                  <p className="mb-0">{content.substring(0, 200)}...</p>
                </Card.Body>
              </Card>
            </div>
            
            <div className="d-flex justify-content-between">
              <Button variant="primary" onClick={handleReset}>
                Check Another Article
              </Button>
              
              <Button
                variant={result.probability > 0.5 ? 'outline-danger' : 'outline-success'}
                disabled
              >
                {result.probability > 0.5
                  ? 'Exercise caution with this content'
                  : 'This content seems trustworthy'}
              </Button>
            </div>
          </Card.Body>
        </Card>
      ) : (
        <Form noValidate validated={validated} onSubmit={handleSubmit}>
          <Form.Group className="mb-3" controlId="newsTitle">
            <Form.Label>News Headline</Form.Label>
            <Form.Control
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
              placeholder="Enter the news headline or title"
              disabled={loading}
            />
            <Form.Control.Feedback type="invalid">
              Please enter a news headline.
            </Form.Control.Feedback>
          </Form.Group>
          
          <Form.Group className="mb-3" controlId="newsContent">
            <Form.Label>News Content</Form.Label>
            <Form.Control
              as="textarea"
              rows={6}
              value={content}
              onChange={(e) => setContent(e.target.value)}
              required
              placeholder="Paste the news article content here"
              disabled={loading}
            />
            <Form.Control.Feedback type="invalid">
              Please enter the news content.
            </Form.Control.Feedback>
          </Form.Group>
          
          <div className="d-grid">
            <Button
              variant="primary"
              type="submit"
              disabled={loading}
              size="lg"
            >
              {loading ? (
                <>
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                  />
                  <span className="ms-2">Analyzing...</span>
                </>
              ) : (
                'Check for Fake News'
              )}
            </Button>
          </div>
        </Form>
      )}
    </Layout>
  );
};

export default FakeNewsCheckPage;