#!/bin/bash

# Load configuration from the root directory
cd ..  # Move to the root directory to load the config file
source chatbot-config.txt
cd aws  # Return to the aws directory

# SSH into the EC2 instance and set up directories
echo "Setting up model directory on EC2..."
ssh -i ../YourKeyPairName.pem ec2-user@$EC2_PUBLIC_IP "sudo mkdir -p /mnt/efs/models"

# For demonstration, we'll create a dummy model file
echo "Creating sample model file..."
mkdir -p temp_models
cat > temp_models/model_info.json << EOF
{
  "model_name": "health-chatbot",
  "description": "A chatbot model trained on health podcast data",
  "version": "1.0.0",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# Generate a dummy model file (replace this with your actual model)
echo "Creating dummy model file for testing..."
cat > temp_models/model.py << EOF
# Dummy model implementation
def predict(text):
    return f"This is a health-related response to: {text}"
EOF

# Upload the models to the EC2 instance
echo "Uploading models to EC2..."
scp -i ../YourKeyPairName.pem -r temp_models/* ec2-user@$EC2_PUBLIC_IP:/mnt/efs/models/

# Verify the upload
echo "Verifying model upload..."
ssh -i ../YourKeyPairName.pem ec2-user@$EC2_PUBLIC_IP "ls -la /mnt/efs/models/"

# Clean up temp files
echo "Cleaning up temporary files..."
rm -rf temp_models

echo "Model upload complete!"
