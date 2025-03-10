import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Form, Button, Card } from 'react-bootstrap';
import axios from 'axios';

const API_URL = "https://your-api-gateway-url.amazonaws.com/prod";

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    testConnection();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const testConnection = async () => {
    try {
      setConnectionStatus('Connecting...');
      // Comment out this block until working backend
      /*
      const response = await axios.get(`${API_URL}/api/test`);
      if (response.data && response.data.status === 'success') {
        setConnectionStatus('Connected');
      } else {
        setConnectionStatus('Connection Error');
      }
      */
      // For now, just set status manually since we don't have a backend yet
      setConnectionStatus('No Backend Yet');
    } catch (error) {
      console.error('Connection test failed:', error);
      setConnectionStatus('Connection Error');
    }
  };

  // Send a message to the chatbot
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to the chat
    const userMessage = { text: input, sender: 'user', timestamp: new Date().toISOString() };
    setMessages([...messages, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Temporarily simulate a response until backend is ready
      setTimeout(() => {
        const botMessage = {
          text: `I'm a sample response. When the backend is connected, I'll provide real health information about: "${input}"`,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, botMessage]);
        setIsLoading(false);
      }, 1000);
      
      // Comment out this block until you have a working backend
      /*
      // Send the message to the API
      const response = await axios.post(`${API_URL}/api/chat`, {
        message: input
      });

      // Add bot response to the chat
      if (response.data && response.data.response) {
        const botMessage = {
          text: response.data.response,
          sender: 'bot',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, botMessage]);
      }
      */
    } catch (error) {
      console.error('Error sending message:', error);
      // Add error message
      const errorMessage = {
        text: 'Sorry, there was an error processing your request.',
        sender: 'bot',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  return (
    <Container className="py-5">
      <Row className="justify-content-center">
        <Col md={8}>
          <Card className="shadow">
            <Card.Header className="bg-primary text-white">
              <h4 className="mb-0">Health Assistant</h4>
              <small className="text-light">Status: {connectionStatus}</small>
            </Card.Header>
            <Card.Body className="chat-container bg-light" style={{ height: '400px', overflowY: 'auto' }}>
              {messages.length === 0 ? (
                <div className="text-center text-muted my-5">
                  <p>Ask me anything about health and wellness!</p>
                </div>
              ) : (
                messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`mb-3 d-flex ${msg.sender === 'user' ? 'justify-content-end' : 'justify-content-start'}`}
                  >
                    <div
                      className={`p-3 rounded-3 ${
                        msg.sender === 'user'
                          ? 'bg-primary text-white'
                          : msg.isError
                          ? 'bg-danger text-white'
                          : 'bg-white border'
                      }`}
                      style={{ maxWidth: '80%' }}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="mb-3 d-flex justify-content-start">
                  <div className="p-3 rounded-3 bg-white border">
                    <div className="spinner-grow spinner-grow-sm text-primary" role="status">
                      <span className="visually-hidden">Loading...</span>
                    </div>{' '}
                    Thinking...
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </Card.Body>
            <Card.Footer className="bg-white">
              <Form onSubmit={sendMessage}>
                <Row>
                  <Col>
                    <Form.Control
                      type="text"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      placeholder="Type your message here..."
                      disabled={isLoading}
                    />
                  </Col>
                  <Col xs="auto">
                    <Button type="submit" variant="primary" disabled={isLoading}>
                      {isLoading ? (
                        <>
                          <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                          <span className="visually-hidden">Loading...</span>
                        </>
                      ) : (
                        'Send'
                      )}
                    </Button>
                  </Col>
                </Row>
              </Form>
            </Card.Footer>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default ChatBot;