#!/bin/bash

# AWS CLI Configuration
echo "Configuring AWS CLI..."
# aws configure  # Uncomment if you need to configure AWS CLI

# Create IAM Role for EC2 with EFS access
echo "Creating IAM Role for EC2..."
aws iam create-role --role-name HealthChatbotEC2Role --assume-role-policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}'

# Attach necessary policies
aws iam attach-role-policy --role-name HealthChatbotEC2Role --policy-arn arn:aws:iam::aws:policy/AmazonEFS-FullAccess
aws iam attach-role-policy --role-name HealthChatbotEC2Role --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess

# Create an instance profile and add the role to it
aws iam create-instance-profile --instance-profile-name HealthChatbotEC2Profile
aws iam add-role-to-instance-profile --instance-profile-name HealthChatbotEC2Profile --role-name HealthChatbotEC2Role

# Create a security group for EC2
echo "Creating security group for EC2..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name HealthChatbotSG --description "Security group for Health Chatbot" --query 'GroupId' --output text)

# Allow inbound SSH traffic
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0

# Allow HTTP and HTTPS traffic
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 443 --cidr 0.0.0.0/0

# Allow traffic on port 5000 (for your API)
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 5000 --cidr 0.0.0.0/0

# Create EFS file system
echo "Creating EFS file system..."
EFS_ID=$(aws efs create-file-system --performance-mode generalPurpose --throughput-mode bursting --encrypted --tags Key=Name,Value=HealthChatbotModels --query 'FileSystemId' --output text)

# Create mount target in the default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[0].SubnetId' --output text)

aws efs create-mount-target --file-system-id $EFS_ID --subnet-id $SUBNET_ID --security-groups $SECURITY_GROUP_ID

# Create EC2 instance
echo "Creating EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t2.medium \
  --key-name YourKeyPairName \
  --security-group-ids $SECURITY_GROUP_ID \
  --subnet-id $SUBNET_ID \
  --iam-instance-profile Name=HealthChatbotEC2Profile \
  --user-data "$(cat << 'EOF'
#!/bin/bash
# Update system
apt-get update -y
apt-get upgrade -y

# Install required packages
apt-get install -y python3-pip python3-venv git amazon-efs-utils

# Create directories
mkdir -p /opt/chatbot
mkdir -p /mnt/efs

# Mount EFS
echo "$EFS_ID:/ /mnt/efs efs defaults,tls 0 0" >> /etc/fstab
mount -a

# Set up Python environment
cd /opt/chatbot
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install flask flask-cors gunicorn boto3 transformers torch

# Create a symbolic link to the models directory
ln -s /mnt/efs/models /opt/chatbot/models

# Create a basic test server
cat > /opt/chatbot/app.py << 'APPEOF'
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'message': 'Hello from Health Chatbot API!',
        'status': 'success'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # This is where you would use your trained model to generate a response
    # For now, we'll just echo the message back
    response = f"You asked about health: {user_message}"
    
    return jsonify({
        'response': response,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
APPEOF

# Create a systemd service file
cat > /etc/systemd/system/chatbot.service << 'SERVICEEOF'
[Unit]
Description=Health Chatbot API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/chatbot
ExecStart=/opt/chatbot/venv/bin/gunicorn -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable and start the service
systemctl enable chatbot
systemctl start chatbot

EOF
)" \
  --query 'Instances[0].InstanceId' \
  --output text)

# Wait for instance to be running
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get the public IP address
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Create API Gateway
echo "Creating API Gateway..."
API_ID=$(aws apigateway create-rest-api --name "ChatbotAPI" --description "API for Health Chatbot" --endpoint-configuration "{ \"types\": [\"REGIONAL\"] }" --query 'id' --output text)

# Get the root resource ID
ROOT_RESOURCE_ID=$(aws apigateway get-resources --rest-api-id $API_ID --query 'items[0].id' --output text)

# Create API resources
API_RESOURCE_ID=$(aws apigateway create-resource --rest-api-id $API_ID --parent-id $ROOT_RESOURCE_ID --path-part "api" --query 'id' --output text)
CHAT_RESOURCE_ID=$(aws apigateway create-resource --rest-api-id $API_ID --parent-id $API_RESOURCE_ID --path-part "chat" --query 'id' --output text)
TEST_RESOURCE_ID=$(aws apigateway create-resource --rest-api-id $API_ID --parent-id $API_RESOURCE_ID --path-part "test" --query 'id' --output text)

# Create methods and integration for the test endpoint
aws apigateway put-method --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method GET --authorization-type NONE
aws apigateway put-integration --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method GET --type HTTP --integration-http-method GET --uri "http://$PUBLIC_IP:5000/api/test" --passthrough-behavior WHEN_NO_MATCH

# Create methods and integration for the chat endpoint
aws apigateway put-method --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method POST --authorization-type NONE
aws apigateway put-integration --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method POST --type HTTP --integration-http-method POST --uri "http://$PUBLIC_IP:5000/api/chat" --passthrough-behavior WHEN_NO_MATCH --request-templates '{"application/json": "{ \"message\": $input.json(\"$.message\") }"}'

# Create method responses
aws apigateway put-method-response --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method GET --status-code 200 --response-models '{"application/json": "Empty"}'
aws apigateway put-method-response --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method POST --status-code 200 --response-models '{"application/json": "Empty"}'

# Create integration responses
aws apigateway put-integration-response --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method GET --status-code 200 --selection-pattern ""
aws apigateway put-integration-response --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method POST --status-code 200 --selection-pattern ""

# Enable CORS
aws apigateway put-method --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method OPTIONS --authorization-type NONE
aws apigateway put-method --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method OPTIONS --authorization-type NONE

aws apigateway put-integration --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method OPTIONS --type MOCK --integration-http-method OPTIONS --request-templates '{"application/json": "{\"statusCode\": 200}"}'
aws apigateway put-integration --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method OPTIONS --type MOCK --integration-http-method OPTIONS --request-templates '{"application/json": "{\"statusCode\": 200}"}'

aws apigateway put-method-response --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Origin": true, "method.response.header.Access-Control-Allow-Methods": true, "method.response.header.Access-Control-Allow-Headers": true}'
aws apigateway put-method-response --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Origin": true, "method.response.header.Access-Control-Allow-Methods": true, "method.response.header.Access-Control-Allow-Headers": true}'

aws apigateway put-integration-response --rest-api-id $API_ID --resource-id $TEST_RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Origin": "'"'*'"'", "method.response.header.Access-Control-Allow-Methods": "'"'GET,OPTIONS'"'", "method.response.header.Access-Control-Allow-Headers": "'"'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"'"}'
aws apigateway put-integration-response --rest-api-id $API_ID --resource-id $CHAT_RESOURCE_ID --http-method OPTIONS --status-code 200 --response-parameters '{"method.response.header.Access-Control-Allow-Origin": "'"'*'"'", "method.response.header.Access-Control-Allow-Methods": "'"'POST,OPTIONS'"'", "method.response.header.Access-Control-Allow-Headers": "'"'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"'"}'

# Deploy the API
aws apigateway create-deployment --rest-api-id $API_ID --stage-name "prod"

# Get the API endpoint URL
API_URL="https://$API_ID.execute-api.$(aws configure get region).amazonaws.com/prod"

echo "Setup completed!"
echo "EC2 Public IP: $PUBLIC_IP"
echo "API Gateway URL: $API_URL"
echo "API Endpoints:"
echo "  - Test: $API_URL/api/test"
echo "  - Chat: $API_URL/api/chat"

# Save configuration for later use
cd ..  # Go back to the project root
cat > chatbot-config.txt << EOF
EC2_INSTANCE_ID=$INSTANCE_ID
EC2_PUBLIC_IP=$PUBLIC_IP
EFS_ID=$EFS_ID
API_GATEWAY_ID=$API_ID
API_URL=$API_URL
EOF

echo "Configuration saved to chatbot-config.txt"
