import React from 'react';
import { Button } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import Layout from '../components/Layout';

const NotFoundPage = () => {
  return (
    <Layout>
      <div className="text-center py-5">
        <h1 className="display-1 fw-bold">404</h1>
        <p className="fs-3">
          <span className="text-danger">Oops!</span> Page not found.
        </p>
        <p className="lead">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Button as={Link} to="/" variant="primary">
          Go Home
        </Button>
      </div>
    </Layout>
  );
};

export default NotFoundPage;