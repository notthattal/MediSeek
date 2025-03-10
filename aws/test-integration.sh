#!/bin/bash

# Load configuration from the root directory
cd ..  # Move to the root directory to load the config file
source chatbot-config.txt
cd aws  # Return to the aws directory

echo "Testing Backend-Frontend Integration"
echo "-----------------------------------"

# Test the API Gateway endpoint
echo "Testing API Gateway Test Endpoint..."
TEST_RESPONSE=$(curl -s "$API_URL/api/test")
echo "Response: $TEST_RESPONSE"

if [[ "$TEST_RESPONSE" == *"success"* ]]; then
  echo "✅ API Gateway Test: SUCCESS"
else
  echo "❌ API Gateway Test: FAILED"
fi

# Test the Chat endpoint
echo -e "\nTesting Chat Endpoint..."
CHAT_RESPONSE=$(curl -s -X POST "$API_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are some tips for better sleep?"}')
echo "Response: $CHAT_RESPONSE"

if [[ "$CHAT_RESPONSE" == *"success"* ]]; then
  echo "✅ Chat Endpoint Test: SUCCESS"
else
  echo "❌ Chat Endpoint Test: FAILED"
fi

echo -e "\nFrontend-Backend Integration"
echo "----------------------------"
echo "To test the full integration, follow these steps:"
echo "1. Update the API URL in your frontend app:"
echo "   - Edit ../frontend/src/components/ChatBot.jsx"
echo "   - Change API_URL to: $API_URL"
echo "2. Start your frontend development server:"
echo "   cd ../frontend && npm run dev"
echo "3. Open the app in your browser and test sending messages"

echo -e "\nWebSocket Tester (if using WebSockets)"
echo "---------------------------------"
cd ..  # Return to root directory to create the HTML file
cat > websocket-test.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Tester</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #messageLog { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        input, button { padding: 8px; margin-right: 5px; }
        .sent { color: blue; }
        .received { color: green; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>WebSocket Connection Tester</h1>
    <div>
        <label for="wsUrl">WebSocket URL:</label>
        <input type="text" id="wsUrl" value="wss://your-websocket-url.amazonaws.com" style="width: 350px;">
        <button onclick="connectWebSocket()">Connect</button>
        <button onclick="disconnectWebSocket()">Disconnect</button>
        <span id="connectionStatus">Disconnected</span>
    </div>
    <div id="messageLog"></div>
    <div>
        <input type="text" id="messageInput" placeholder="Type a message..." style="width: 70%;">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        let socket;
        const messageLog = document.getElementById('messageLog');
        const connectionStatus = document.getElementById('connectionStatus');
        
        function logMessage(message, type) {
            const entry = document.createElement('div');
            entry.className = type;
            entry.textContent = message;
            messageLog.appendChild(entry);
            messageLog.scrollTop = messageLog.scrollHeight;
        }
        
        function connectWebSocket() {
            const url = document.getElementById('wsUrl').value;
            try {
                socket = new WebSocket(url);
                connectionStatus.textContent = 'Connecting...';
                
                socket.onopen = function(e) {
                    connectionStatus.textContent = 'Connected';
                    logMessage('Connection established', 'info');
                };
                
                socket.onmessage = function(event) {
                    logMessage('Received: ' + event.data, 'received');
                };
                
                socket.onclose = function(event) {
                    if (event.wasClean) {
                        logMessage('Connection closed cleanly, code=' + event.code + ' reason=' + event.reason, 'info');
                    } else {
                        logMessage('Connection died', 'error');
                    }
                    connectionStatus.textContent = 'Disconnected';
                };
                
                socket.onerror = function(error) {
                    logMessage('Error: ' + error.message, 'error');
                    connectionStatus.textContent = 'Error';
                };
            } catch (err) {
                logMessage('Error: ' + err.message, 'error');
            }
        }
        
        function disconnectWebSocket() {
            if (socket) {
                socket.close();
                socket = null;
            }
        }
        
        function sendMessage() {
            const messageInput = document.getElementById('messageInput');
            const message = messageInput.value;
            
            if (!socket || socket.readyState !== WebSocket.OPEN) {
                logMessage('Not connected', 'error');
                return;
            }
            
            if (!message) {
                return;
            }
            
            socket.send(message);
            logMessage('Sent: ' + message, 'sent');
            messageInput.value = '';
        }
    </script>
</body>
</html>
EOF

echo "WebSocket test HTML created in the root directory."
cd aws  # Return to aws directory
