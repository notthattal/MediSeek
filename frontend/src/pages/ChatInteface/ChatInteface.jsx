"use client"
import { useState, useEffect, useRef, useCallback } from "react"
import MessageList from "../../components/MessageList/MessageList"
import ChatInput from "../../components/ChatInput/ChatInput"
import styles from "./ChatInterface.module.css"

// websocket service
const createWebSocketService = () => {
  // singleton instance
  let serviceInstance = null;

  // debug logging
  const DEBUG = true;

  class WebSocketService {
    constructor() {
      // if instance exists, return it
      if (serviceInstance) {
        if (DEBUG) console.log('returning existing websocket service instance');
        return serviceInstance;
      }

      // set url
      this.url = "wss://bvwq1y85ha.execute-api.us-east-1.amazonaws.com/prod";
      this.socket = null;
      this.isConnected = false;
      this.messageCallbacks = [];
      this.connectionCallbacks = [];
      this.errorCallbacks = [];
      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 5;
      this.reconnectDelay = 3000;
      // 3s delay
      this.pendingMessages = [];

      // store instance
      serviceInstance = this;

      // log url
      if (DEBUG) console.log('websocket will connect to:', this.url);

      // setup unload listener
      if (typeof window !== 'undefined') {
        // close socket on unload
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
          this.reconnectAttempts = 0;
          // reset reconnect attempts
          this.reconnectDelay = 3000;
          // reset delay

          // notify callbacks
          this.connectionCallbacks.forEach(callback => callback(true));

          // send pending msgs
          if (this.pendingMessages.length > 0) {
            if (DEBUG) console.log(`sending ${this.pendingMessages.length} pending messages`);
            [...this.pendingMessages].forEach(item => {
              this.doSendMessage(item.message, item.modelType);
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
            // send raw data on parse fail
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

          // no reconnect on normal close or max attempts
          if (event.code !== 1000 && event.code !== 1001 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(this.reconnectDelay * this.reconnectAttempts, 30000);
            // max 30s
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

    sendMessage(message, modelType = "deepseek") {
      if (DEBUG) console.log(`attempting to send message using ${modelType} model:`, message);
      console.log('WebSocket ready state:', this.socket?.readyState);
      console.log('WebSocket connected status:', this.isConnected);

      if (!this.isConnected) {
        if (DEBUG) console.log('websocket not connected, queuing message and connecting...');
        // queue message
        if (!this.pendingMessages.some(item => item.message === message)) {
          this.pendingMessages.push({ message, modelType });
        }
        this.connect();
        return;
      }

      this.doSendMessage(message, modelType);
    }

    doSendMessage(message, modelType = "deepseek") {
      console.log('doSendMessage called, socket state:', this.socket?.readyState);

      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        const payload = JSON.stringify({
          action: 'sendMessage',
          message,
          model_type: modelType
        });
        if (DEBUG) console.log('sending websocket message:', payload);
        this.socket.send(payload);
      } else {
        console.error('websocket is not connected or ready. readystate:', this.socket ? this.socket.readyState : 'no socket');
        // queue message if needed
        if (!this.pendingMessages.some(item => item.message === message)) {
          this.pendingMessages.push({ message, modelType });
        }

        // reconnect if needed
        if (this.socket && (this.socket.readyState === WebSocket.CLOSING || this.socket.readyState === WebSocket.CLOSED)) {
          if (DEBUG) console.log('socket is closing or closed, attempting to reconnect...');
          this.connect();
        }
      }
    }

    onMessage(callback) {
      // dedupe callbacks
      this.messageCallbacks = this.messageCallbacks.filter(cb => cb !== callback);
      this.messageCallbacks.push(callback);
    }

    onConnectionChange(callback) {
      // dedupe callbacks
      this.connectionCallbacks = this.connectionCallbacks.filter(cb => cb !== callback);
      this.connectionCallbacks.push(callback);

      // notify current state
      callback(this.isConnected);
    }

    onError(callback) {
      // dedupe callbacks
      this.errorCallbacks = this.errorCallbacks.filter(cb => cb !== callback);
      this.errorCallbacks.push(callback);
    }
  }

  if (DEBUG) console.log('websocket service created');
  return new WebSocketService();
};

// create websocket instance
const webSocketService = createWebSocketService();

const ChatInterface = () => {
  console.log('chatbot component loaded');

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('Connecting...');
  const [hasSetupWebSocket, setHasSetupWebSocket] = useState(false);
  // default: deepseek model
  const [selectedModel, setSelectedModel] = useState('deepseek');
  const fallbackTimerRef = useRef(null);

  // setup websocket handlers
  const setupWebSocketHandlers = useCallback(() => {
    if (hasSetupWebSocket) return;

    console.log('setting up websocket handlers');

    webSocketService.onConnectionChange((isConnected) => {
      console.log('websocket connection status changed:', isConnected);
      setConnectionStatus(isConnected ? 'Connected' : 'Disconnected');
    });

    webSocketService.onMessage((data) => {
      console.log('message received from websocket:', data);

      // clear fallback timer
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }

      // if response exists
      if (data && data.response) {
        const aiMessage = {
          id: Date.now() + 1,
          content: data.response,
          role: "assistant",
        }
        setMessages(prevMessages => [...prevMessages, aiMessage]);
        setIsLoading(false);
      }
      // if error exists
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
      // if internal error
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
      // if raw data exists
      else if (data && data.raw) {
        try {
          // parse raw json
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
          console.log('failed to parse raw data as json');
        }

        const botMessage = {
          id: Date.now() + 1,
          content: "Received a response in an unexpected format.",
          role: "assistant",
        };
        setMessages(prevMessages => [...prevMessages, botMessage]);
        setIsLoading(false);

        // log raw data
        console.warn("raw data received:", data.raw);
      }
      // fallback for unknown format
      else {
        console.warn("received unexpected message format:", data);
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
      console.error('websocket error:', error);
      setConnectionStatus('Connection Error');

      // add error msg to chat
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

  // connect on mount
  useEffect(() => {
    console.log('chatinterface component mounted, initializing websocket');

    // connect if needed
    if (!webSocketService.isConnected) {
      webSocketService.connect();
    }

    // setup handlers
    setupWebSocketHandlers();

    // keep websocket alive
    return () => {
      console.log('chatinterface component unmounting, but keeping websocket alive');

      // clear fallback timer on unmount
      if (fallbackTimerRef.current) {
        clearTimeout(fallbackTimerRef.current);
        fallbackTimerRef.current = null;
      }
    };
  }, [setupWebSocketHandlers]);

  const handleSendMessage = (messageText) => {
    if (!messageText.trim()) return;

    // create user msg
    const userMessage = {
      id: Date.now(),
      content: messageText,
      role: "user",
    };

    // add msg to chat
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsLoading(true);

    // send msg via websocket
    console.log(`sending message via websocket using ${selectedModel} model:`, messageText);
    webSocketService.sendMessage(messageText, selectedModel);

    // set fallback timer
    fallbackTimerRef.current = setTimeout(() => {
      if (isLoading) {
        console.log('no websocket response received within timeout, using fallback');
        const aiMessage = {
          id: Date.now() + 1,
          content: `I apologize for the delay. It seems like the server is taking longer than expected to respond. Please try again in a moment.`,
          role: "assistant",
          isWarning: true
        };
        setMessages(prevMessages => [...prevMessages, aiMessage]);
        setIsLoading(false);
      }
    }, 20000);
  };

  // toggle model
  const toggleModel = () => {
    setSelectedModel(prevModel => prevModel === "deepseek" ? "lstm" : "deepseek");
  };

  return (
    <div className={styles.container}>
      <div className={styles.chatContainer}>
        <h1 className={styles.heading}>What can I help with?</h1>
        <div className={styles.statusInfo}>
          <div className={styles.connectionStatus}>Status: {connectionStatus}</div>
          <div className={styles.modelSelector}>
            <label>
              <input
                type="radio"
                checked={selectedModel === "deepseek"}
                onChange={() => setSelectedModel("deepseek")}
              />
              DeepSeek Model
            </label>
            <label>
              <input
                type="radio"
                checked={selectedModel === "lstm"}
                onChange={() => setSelectedModel("lstm")}
              />
              LSTM Model
            </label>
          </div>
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
            <button className={styles.actionButton} onClick={toggleModel}>
              <span className={styles.icon}>🔄</span>
              Switch Model ({selectedModel})
            </button>
            <button className={styles.actionButton}>
              <span className={styles.icon}>📝</span>
              Summarize text
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
