#!/bin/bash

# Install Node.js and npm if not already installed
# This assumes you're on a Debian/Ubuntu-based system
# Adjust accordingly for other systems
if ! command -v node &> /dev/null; then
    echo "Installing Node.js and npm..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Create a new Vite project with React in the frontend directory
echo "Creating a new Vite + React project..."
npm create vite@latest frontend -- --template react

# Navigate to the project directory
cd frontend

# Install dependencies
echo "Installing dependencies..."
npm install bootstrap react-bootstrap axios

# Create component directories
mkdir -p src/components

# Create a ChatBot component
cat > src/components/ChatBot.jsx << 'EOF'
import React, { useState, useEffect, useRef } from 'react';
import { Container, Row, Col, Form, Button, Card } from 'react-bootstrap';
import axios from 'axios';

// Replace with your actual API URL
const API_URL = "https://your-api-gateway-url.amazonaws.com/prod";

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const messagesEndRef = useRef(null);

  // Test the connection to the backend on component mount
  useEffect(() => {
    testConnection();
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Test the connection to the backend
  const testConnection = async () => {
    try {
      setConnectionStatus('Connecting...');
      const response = await axios.get(`${API_URL}/api/test`);
      if (response.data && response.data.status === 'success') {
        setConnectionStatus('Connected');
      } else {
        setConnectionStatus('Connection Error');
      }
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
    } finally {
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
EOF

# Update App.jsx
cat > src/App.jsx << 'EOF'
import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import ChatBot from './components/ChatBot';

function App() {
  return (
    <div className="App">
      <nav className="navbar navbar-dark bg-dark">
        <div className="container">
          <span className="navbar-brand mb-0 h1">Health & Wellness AI</span>
        </div>
      </nav>
      <main>
        <ChatBot />
      </main>
      <footer className="bg-light text-center text-muted py-3 mt-5">
        <div className="container">
          <p>Powered by AI trained on health and wellness data</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
EOF

# Create a custom CSS file
cat > src/App.css << 'EOF'
.chat-container {
  scrollbar-width: thin;
  scrollbar-color: #ccc transparent;
}

.chat-container::-webkit-scrollbar {
  width: 6px;
}

.chat-container::-webkit-scrollbar-thumb {
  background-color: #ccc;
  border-radius: 10px;
}

.chat-container::-webkit-scrollbar-track {
  background: transparent;
}
EOF

# Create .env file with the API URL
# You need to replace this with your actual API Gateway URL
echo "Creating .env file..."
cat > .env << 'EOF'
VITE_API_URL=https://your-api-gateway-url.amazonaws.com/prod
EOF

# Create deployment script
cat > deploy-to-hostgator.sh << 'EOF'
#!/bin/bash

# Build the project
npm run build

# Compress the build directory
echo "Compressing build directory..."
zip -r build.zip dist/

# Upload to Hostgator via FTP
# You'll need to replace these with your actual Hostgator credentials
echo "Uploading to Hostgator..."
echo "This is a placeholder. Please update with your actual Hostgator credentials and path."
echo "Example FTP command:"
echo "ftp -n <<EOF"
echo "open ftp.yourdomain.com"
echo "user yourusername yourpassword"
echo "cd public_html"
echo "binary"
echo "put build.zip"
echo "bye"
echo "EOF"
echo ""
echo "After uploading, you need to extract the zip file on the server using cPanel File Manager or SSH."

chmod +x deploy-to-hostgator.sh

echo "Frontend project created successfully!"
echo "Steps to complete setup:"
echo "1. Navigate to the project directory: cd frontend"
echo "2. Update the API_URL in src/components/ChatBot.jsx with your actual API Gateway URL"
echo "3. Update the .env file with your actual API Gateway URL"
echo "4. Start the development server: npm run dev"
echo "5. When ready to deploy, update the deploy-to-hostgator.sh script with your Hostgator credentials"
echo "6. Run the deployment script: ./deploy-to-hostgator.sh"
