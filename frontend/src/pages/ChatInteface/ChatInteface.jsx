"use client"
import { useState, useEffect, useRef, useCallback } from "react"
import MessageList from "../../components/MessageList/MessageList"
import ChatInput from "../../components/ChatInput/ChatInput"
import styles from "./ChatInterface.module.css"

// WebSocket Service
const createWebSocketService = () => {
  // Create a singleton instance that persists across hot reloads
  let serviceInstance = null;

  // debug mode for additional logging
  const DEBUG = true;

  class WebSocketService {
    constructor() {
      // if we already have an instance, return it
      if (serviceInstance) {
        if (DEBUG) console.log('returning existing websocket service instance');
        return serviceInstance;
      }
      
      // remove trailing slash from url if present
      this.url = "wss://bvwq1y85ha.execute-api.us-east-1.amazonaws.com/prod";
      this.socket = null;
      this.isConnected = false;
      this.messageCallbacks = [];
      this.connectionCallbacks = [];
      this.errorCallbacks = [];
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectDelay = 3000; // start with 3 seconds
      this.pendingMessages = [];
      
      // store this instance
      serviceInstance = this;
      
      // log the url we're attempting to connect to
      if (DEBUG) console.log('websocket will connect to:', this.url);
      
      // setup window event listeners to handle page lifecycle
      if (typeof window !== 'undefined') {
        // try to gracefully close the connection when the page is unloaded
        window.addEventListener('beforeunload', () => {
          if (this.socket && this.isConnected) {
            if (DEBUG) console.log('page is being unloaded, closing websocket');
            this.socket.close(1000, "page unload");
          }
        });
      }
    }

    connect() {
      if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
        if (DEBUG) console.log('websocket already connected or connecting, skipping new connection');
        return;
      }
      
      if (DEBUG) console.log('attempting to connect to websocket...');
      try {
        this.socket = new WebSocket(this.url);
        
        this.socket.onopen = () => {
          if (DEBUG) console.log('websocket connection established successfully');
          this.isConnected = true;
          this.reconnectAttempts = 0; // reset reconnect attempts on successful connection
          this.reconnectDelay = 3000; // reset reconnect delay
          
          // notify all connection callbacks
          this.connectionCallbacks.forEach(callback => callback(true));
          
          // send any pending messages
          if (this.pendingMessages.length > 0) {
            if (DEBUG) console.log(`sending ${this.pendingMessages.length} pending messages`);
            [...this.pendingMessages].forEach(message => {
              this.doSendMessage(message);
            });
            this.pendingMessages = [];
          }
        };
        
        this.socket.onmessage = (event) => {
          if (DEBUG) console.log('websocket raw message received:', event.data);
          try {
            const data = JSON.parse(event.data);
            if (DEBUG) console.log('parsed websocket message:', data);
            this.messageCallbacks.forEach(callback => callback(data));
          } catch (error) {
            console.error('error parsing websocket message:', error);
            // try to send the raw data even if parsing failed
            this.messageCallbacks.forEach(callback => callback({ 
              error: "failed to parse response", 
              raw: event.data 
            }));
          }
        };
        
        this.socket.onclose = (event) => {
          if (DEBUG) console.log('websocket disconnected. code:', event.code, 'reason:', event.reason);
          this.isConnected = false;
          this.connectionCallbacks.forEach(callback => callback(false));
          
          // don't reconnect if the connection was closed normally (1000 = normal closure, 1001 = going away)
          // or if max attempts reached
          if (event.code !== 1000 && event.code !== 1001 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(this.reconnectDelay * this.reconnectAttempts, 30000); // max 30 seconds
            if (DEBUG) console.log(`will attempt to reconnect in ${delay/1000} seconds... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), delay);
          } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('maximum reconnect attempts reached. please refresh the page.');
            this.errorCallbacks.forEach(callback => 
              callback(new Error('maximum reconnect attempts reached. please refresh the page.'))
            );
          }
        };
        
        this.socket.onerror = (error) => {
          console.error('websocket error occurred:', error);
          this.errorCallbacks.forEach(callback => callback(error));
        };
      } catch (err) {
        console.error('error creating websocket connection:', err);
        this.errorCallbacks.forEach(callback => callback(err));
      }
    }
    
    disconnect() {
      if (this.socket && this.isConnected) {
        if (DEBUG) console.log('manually disconnecting websocket');
        this.socket.close(1000, "user initiated disconnect");
        this.isConnected = false;
      }
    }
    
    sendMessage(message) {
      if (DEBUG) console.log('attempting to send message:', message);
      console.log('WebSocket ready state:', this.socket?.readyState);
      console.log('WebSocket connected status:', this.isConnected);
      
      if (!this.isConnected) {
        if (DEBUG) console.log('websocket not connected, queuing message and connecting...');
        // store message to send after connection
        if (!this.pendingMessages.includes(message)) {
          this.pendingMessages.push(message);
        }
        this.connect();
        return;
      }
      
      this.doSendMessage(message);
    }
    
    doSendMessage(message) {
      console.log('doSendMessage called, socket state:', this.socket?.readyState);
      
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        const payload = JSON.stringify({
          action: 'sendMessage',
          message
        });
        if (DEBUG) console.log('sending websocket message:', payload);
        this.socket.send(payload);
      } else {
        console.error('websocket is not connected or ready. readystate:', this.socket ? this.socket.readyState : 'no socket');
        // queue the message for later sending if not already queued
        if (!this.pendingMessages.includes(message)) {
          this.pendingMessages.push(message);
        }
        
        // if the socket exists but is closing or closed, reconnect
        if (this.socket && (this.socket.readyState === WebSocket.CLOSING || this.socket.readyState === WebSocket.CLOSED)) {
          if (DEBUG) console.log('socket is closing or closed, attempting to reconnect...');
          this.connect();
        }
      }
    }
    
    onMessage(callback) {
      // remove any duplicate callbacks
      this.messageCallbacks = this.messageCallbacks.filter(cb => cb !== callback);
      this.messageCallbacks.push(callback);
    }
    
    onConnectionChange(callback) {
      // remove any duplicate callbacks
      this.connectionCallbacks = this.connectionCallbacks.filter(cb => cb !== callback);
      this.connectionCallbacks.push(callback);
      
      // immediately notify of current state
      callback(this.isConnected);
    }
    
    onError(callback) {
      // remove any duplicate callbacks
      this.errorCallbacks = this.errorCallbacks.filter(cb => cb !== callback);
      this.errorCallbacks.push(callback);
    }
  }

  if (DEBUG) console.log('websocket service created');
  return new WebSocketService();
};

// Create a single instance of the WebSocket service
const webSocketService = createWebSocketService();

const ChatInterface = () => {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState('Connecting...')
  const [hasSetupWebSocket, setHasSetupWebSocket] = useState(false)
  const fallbackTimerRef = useRef(null)

  // Setup WebSocket event handlers
  const setupWebSocketHandlers = useCallback(() => {
    if (hasSetupWebSocket) return;
    
    console.log('Setting up WebSocket handlers');
    
    webSocketService.onConnectionChange((isConnected) => {
      console.log('WebSocket connection status changed:', isConnected);
      setConnectionStatus(isConnected ? 'Connected' : 'Disconnected');
    });
    
    webSocketService.onMessage((data) => {
      console.log('Message received from WebSocket:', data);
      
      // Clear any pending fallback timer
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }
      
      // Check for response field
      if (data && data.response) {
        const aiMessage = {
          id: Date.now() + 1,
          content: data.response,
          role: "assistant",
        }
        setMessages(prevMessages => [...prevMessages, aiMessage]);
        setIsLoading(false);
      } 
      // Check for error field
      else if (data && data.error) {
        const errorMessage = {
          id: Date.now() + 1,
          content: `Error: ${data.error}`,
          role: "assistant",
          isError: true
        };
        setMessages(prevMessages => [...prevMessages, errorMessage]);
        setIsLoading(false);
      }
      // Handle internal server error message format
      else if (data && data.message === "Internal server error") {
        const errorMessage = {
          id: Date.now() + 1,
          content: "The server encountered an error processing your request. Please try again later.",
          role: "assistant",
          isError: true
        };
        setMessages(prevMessages => [...prevMessages, errorMessage]);
        setIsLoading(false);
      }
      // Handle the raw data format (if JSON parsing failed)
      else if (data && data.raw) {
        try {
          // Try to parse the raw data as JSON
          const rawData = JSON.parse(data.raw);
          if (rawData && rawData.response) {
            const aiMessage = {
              id: Date.now() + 1,
              content: rawData.response,
              role: "assistant",
            };
            setMessages(prevMessages => [...prevMessages, aiMessage]);
            setIsLoading(false);
            return;
          }
        } catch (e) {
          console.log('Failed to parse raw data as JSON');
        }
        
        const botMessage = {
          id: Date.now() + 1,
          content: "Received a response in an unexpected format.",
          role: "assistant",
        };
        setMessages(prevMessages => [...prevMessages, botMessage]);
        setIsLoading(false);
        
        // Log the raw data for debugging
        console.warn("Raw data received:", data.raw);
      }
      // Fallback for unexpected message format
      else {
        console.warn("Received unexpected message format:", data);
        const botMessage = {
          id: Date.now() + 1,
          content: "Received a response in an unexpected format.",
          role: "assistant",
        };
        setMessages(prevMessages => [...prevMessages, botMessage]);
        setIsLoading(false);
      }
    });
    
    webSocketService.onError((error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('Connection Error');
      
      // Add an error message to the chat
      const errorMessage = {
        id: Date.now() + 1,
        content: typeof error === 'string' ? error : 'Connection error occurred. Please try again later.',
        role: "assistant",
        isError: true
      };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
      setIsLoading(false);
    });
    
    setHasSetupWebSocket(true);
  }, [hasSetupWebSocket]);

  // Connect to WebSocket on component mount
  useEffect(() => {
    console.log('ChatInterface component mounted, initializing WebSocket');
    
    // Connect to WebSocket if not already connected
    if (!webSocketService.isConnected) {
      webSocketService.connect();
    }
    
    // Setup handlers
    setupWebSocketHandlers();
    
    // No need to disconnect on unmount - we want the WebSocket to stay alive
    return () => {
      console.log('ChatInterface component unmounting, but keeping WebSocket alive');
      
      // Clear any pending fallback timer on unmount
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }
    };
  }, [setupWebSocketHandlers]);
  
  const handleSendMessage = (messageText) => {
    if (!messageText.trim()) return
    
    // Create user message with the expected format for your MessageList component
    const userMessage = {
      id: Date.now(),
      content: messageText,
      role: "user",
    }
    
    // Add user message to chat
    setMessages(prevMessages => [...prevMessages, userMessage])
    setIsLoading(true)
    
    // Send message via WebSocket
    console.log('Sending message via WebSocket:', messageText)
    webSocketService.sendMessage(messageText)
    
    // Set a fallback timer in case WebSocket response doesn't come back
    fallbackTimerRef.current = setTimeout(() => {
      if (isLoading) {  // Only do this if we haven't received a real response
        console.log('No WebSocket response received within timeout, using fallback')
        const aiMessage = {
          id: Date.now() + 1,
          content: `I apologize for the delay. It seems like the server is taking longer than expected to respond. Please try again in a moment.`,
          role: "assistant",
          isWarning: true
        }
        setMessages(prevMessages => [...prevMessages, aiMessage])
        setIsLoading(false)
      }
    }, 15000) // 15 second timeout
  }

  return (
    <div className={styles.container}>
      <div className={styles.chatContainer}>
        <h1 className={styles.heading}>What can I help with?</h1>
        <div className={styles.statusIndicator}>
          Connection Status: {connectionStatus}
        </div>
        <div className={styles.messageArea}>
          <MessageList messages={messages} isLoading={isLoading} />
        </div>
        <div className={styles.inputArea}>
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
          <div className={styles.quickActions}>
            <button className={styles.actionButton}>
              <span className={styles.icon}>🎨</span>
              Create image
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>✍️</span>
              Help me write
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>🎲</span>
              Surprise me
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>📝</span>
              Summarize text
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ChatInterface