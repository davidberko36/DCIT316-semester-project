import React from 'react';
import { Card, Badge, Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';

const NewsCard = ({ article, showFakeNewsProbability = false }) => {
  // Truncate text to a specific length
  const truncate = (text, maxLength = 100) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  // Format date to a readable format
  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  return (
    <Card className="h-100 shadow-sm">
      {article.urlToImage && (
        <Card.Img 
          variant="top" 
          src={article.urlToImage}
          alt={article.title}
          style={{ height: '180px', objectFit: 'cover' }}
          onError={(e) => {
            e.target.src = 'https://via.placeholder.com/300x180?text=No+Image+Available';
          }}
        />
      )}
      <Card.Body className="d-flex flex-column">
        <div className="d-flex justify-content-between align-items-center mb-2">
          <Badge bg="primary">{article.source?.name || 'Unknown Source'}</Badge>
          <small className="text-muted">{formatDate(article.publishedAt)}</small>
        </div>
        
        <Card.Title>{article.title}</Card.Title>
        <Card.Text>{truncate(article.description, 150)}</Card.Text>
        
        {showFakeNewsProbability && article.fakeProbability !== undefined && (
          <div className="mb-3">
            <Badge bg={article.fakeProbability > 0.5 ? 'danger' : 'success'}>
              {article.fakeProbability > 0.5 
                ? `Likely Fake (${Math.round(article.fakeProbability * 100)}%)` 
                : `Likely Real (${Math.round((1 - article.fakeProbability) * 100)}%)`}
            </Badge>
          </div>
        )}
        
        <div className="mt-auto">
          <Button 
            as={Link} 
            to={`/news/${article.id || encodeURIComponent(article.url)}`} 
            variant="outline-primary" 
            className="w-100"
          >
            Read More
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
};

export default NewsCard;