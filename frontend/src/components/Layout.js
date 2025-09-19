import React from 'react';
import { Container } from 'react-bootstrap';
import Header from './Header';
import Footer from './Footer';

const Layout = ({ children }) => {
  return (
    <div className="d-flex flex-column min-vh-100">
      <Header />
      <Container className="py-4 flex-grow-1">
        {children}
      </Container>
      <Footer />
    </div>
  );
};

export default Layout;