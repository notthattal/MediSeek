import json
import os
import logging
import requests

# setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    # extract connection info
    connection_id = event.get("requestContext", {}).get("connectionId", "unknown")
    route_key = event.get("requestContext", {}).get("routeKey", "$default")
    domain_name = event.get("requestContext", {}).get("domainName")
    stage = event.get("requestContext", {}).get("stage")
    
    logger.info(f"websocket event - route: {route_key}, connection id: {connection_id}")
    logger.info(f"domain: {domain_name}, stage: {stage}")
    
    if route_key == "$connect":
        logger.info(f"new connection: {connection_id}")
        return {"statusCode": 200, "body": "Connected"}
        
    elif route_key == "$disconnect":
        logger.info(f"connection closed: {connection_id}")
        return {"statusCode": 200, "body": "Disconnected"}
        
    elif route_key == "sendMessage":
        try:
            # parse the message
            body = json.loads(event.get("body", "{}"))
            message = body.get("message", "")
            # expect model headeers being "lstm", "deepseek", "mediseek"
            model_type = body.get("model_type", "deepseek")
            
            # EC2 endpoint, change your endpoint ip or change the lambda env varibles
            ec2_ip = os.environ.get("EC2_IP_ADDRESS", "###.###.###")
            ec2_endpoint = f"http://{ec2_ip}:5000/predict"
            
            logger.info(f"sending to EC2 using {model_type} model: {message}")
            
            response = requests.post(
                ec2_endpoint,
                json={"message": message, "model_type": model_type},
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    model_response = response.json()
                    raw_response = model_response.get("response", "")
                    return {
                        "statusCode": 200,
                        "body": json.dumps({"response": raw_response})
                    }
                except json.JSONDecodeError as je:
                    error_message = f"invalid json from server: {str(je)}"
                    logger.error(error_message)
                    return {
                        "statusCode": 500,
                        "body": json.dumps({"error": error_message})
                    }
            else:
                error_message = f"EC2 error: {response.status_code}"
                logger.error(error_message)
                return {
                    "statusCode": 500,
                    "body": json.dumps({"error": error_message})
                }
                
        except Exception as e:
            error_message = f"error: {str(e)}"
            logger.error(error_message)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": error_message})
            }
    
    else:
        logger.warning(f"unknown route: {route_key}")
        return {"statusCode": 400, "body": json.dumps({"error": "unknown route"})}