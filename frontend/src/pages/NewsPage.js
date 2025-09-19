import React, { useState, useEffect } from 'react';
import { Row, Col, Form, Button, Alert, Spinner, Pagination } from 'react-bootstrap';
import Layout from '../components/Layout';
import NewsCard from '../components/NewsCard';
import newsService from '../services/newsService';

const NewsPage = () => {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const articlesPerPage = 9; // Number of articles per page

  const fetchNews = async (page = 1, search = '') => {
    try {
      setLoading(true);
      const params = {
        page,
        pageSize: articlesPerPage,
        ...(search && { q: search })
      };
      
      const response = await newsService.getNews(params);
      
      // Handle different possible response structures
      if (Array.isArray(response)) {
        setNews(response);
        setTotalPages(Math.ceil(response.length / articlesPerPage));
      } else if (response.articles && Array.isArray(response.articles)) {
        setNews(response.articles);
        setTotalPages(Math.ceil((response.totalResults || response.articles.length) / articlesPerPage));
      } else {
        setNews([]);
        setTotalPages(1);
        console.warn('Unexpected response format:', response);
      }
      
      setError(null);
    } catch (err) {
      console.error('Error fetching news:', err);
      setError('Failed to load news articles. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews(currentPage, searchTerm);
  }, [currentPage]);

  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1); // Reset to first page on new search
    fetchNews(1, searchTerm);
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
    window.scrollTo(0, 0);
  };

  // Generate pagination items
  const renderPaginationItems = () => {
    const items = [];
    const maxPagesToShow = 5;
    
    let startPage = Math.max(1, currentPage - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);
    
    if (endPage - startPage + 1 < maxPagesToShow) {
      startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }
    
    // Previous button
    items.push(
      <Pagination.Prev 
        key="prev" 
        onClick={() => handlePageChange(currentPage - 1)}
        disabled={currentPage === 1}
      />
    );
    
    // First page
    if (startPage > 1) {
      items.push(
        <Pagination.Item key={1} onClick={() => handlePageChange(1)}>
          1
        </Pagination.Item>
      );
      if (startPage > 2) {
        items.push(<Pagination.Ellipsis key="ellipsis1" />);
      }
    }
    
    // Page numbers
    for (let page = startPage; page <= endPage; page++) {
      items.push(
        <Pagination.Item 
          key={page} 
          active={page === currentPage}
          onClick={() => handlePageChange(page)}
        >
          {page}
        </Pagination.Item>
      );
    }
    
    // Last page
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) {
        items.push(<Pagination.Ellipsis key="ellipsis2" />);
      }
      items.push(
        <Pagination.Item 
          key={totalPages} 
          onClick={() => handlePageChange(totalPages)}
        >
          {totalPages}
        </Pagination.Item>
      );
    }
    
    // Next button
    items.push(
      <Pagination.Next 
        key="next" 
        onClick={() => handlePageChange(currentPage + 1)}
        disabled={currentPage === totalPages}
      />
    );
    
    return items;
  };

  return (
    <Layout>
      <h1 className="mb-4">Ghana News</h1>
      
      <Form onSubmit={handleSearch} className="mb-4">
        <Row>
          <Col md={8} lg={9}>
            <Form.Group controlId="searchTerm">
              <Form.Control
                type="text"
                placeholder="Search for news..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </Form.Group>
          </Col>
          <Col md={4} lg={3}>
            <Button variant="primary" type="submit" className="w-100">
              Search
            </Button>
          </Col>
        </Row>
      </Form>
      
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
        <>
          <Row xs={1} md={2} lg={3} className="g-4">
            {news.map((article, index) => (
              <Col key={article.id || `${article.url}-${index}`}>
                <NewsCard article={article} />
              </Col>
            ))}
            
            {news.length === 0 && !loading && !error && (
              <Col xs={12}>
                <Alert variant="info">
                  No news articles found. Try a different search term.
                </Alert>
              </Col>
            )}
          </Row>
          
          {totalPages > 1 && (
            <Row className="mt-4">
              <Col className="d-flex justify-content-center">
                <Pagination>{renderPaginationItems()}</Pagination>
              </Col>
            </Row>
          )}
        </>
      )}
    </Layout>
  );
};

export default NewsPage;