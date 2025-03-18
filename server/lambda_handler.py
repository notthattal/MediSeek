import json
import os
import logging
import boto3
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
    
    # log basic info
    logger.info(f"websocket event - route: {route_key}, connection id: {connection_id}")
    logger.info(f"domain: {domain_name}, stage: {stage}")
    
    # handle route keys
    if route_key == "$connect":
        # handle new connection
        logger.info(f"new connection: {connection_id}")
        return {"statusCode": 200, "body": "Connected"}
        
    elif route_key == "$disconnect":
        # handle disconnect
        logger.info(f"connection closed: {connection_id}")
        return {"statusCode": 200, "body": "Disconnected"}
        
    elif route_key == "sendMessage":
        # handle message
        try:
            # parse message
            body = json.loads(event.get("body", "{}"))
            message = body.get("message", "")
            
            # get ec2 endpoint
            ec2_ip = os.environ.get("EC2_IP_ADDRESS", "##.###.##.###")
            ec2_endpoint = f"http://{ec2_ip}:5000/predict"
            
            logger.info(f"sending to ec2: {message}")
            
            # send to ec2
            response = requests.post(
                ec2_endpoint,
                json={"message": message},
                timeout=30
            )
            
            # process response
            if response.status_code == 200:
                try:
                    model_response = response.json()
                    logger.info(f"got model response")
                    
                    # get raw response
                    raw_response = model_response.get("response", "")
                    
                    # return directly
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
                # handle ec2 error
                error_message = f"ec2 error: {response.status_code}"
                logger.error(error_message)
                return {
                    "statusCode": 500,
                    "body": json.dumps({"error": error_message})
                }
                
        except Exception as e:
            # handle exceptions
            error_message = f"error: {str(e)}"
            logger.error(error_message)
            return {
                "statusCode": 500,
                "body": json.dumps({"error": error_message})
            }
    
    else:
        # unknown route
        logger.warning(f"unknown route: {route_key}")
        return {"statusCode": 400, "body": json.dumps({"error": "unknown route"})}