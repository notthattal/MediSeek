from flask import Flask, request, jsonify
import json
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# init flask app
app = Flask(__name__)

# see if cuda is availble
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available: {cuda_available}")

if cuda_available:
    # this outputs the GPU info
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        logger.info(f"GPU {i}: {device_name}")

# global variables for tokenizer and models
model = None
tokenizer = None

try:
    # choose model to laod
    MODEL_SIZE = "1.3b"  
    
    if MODEL_SIZE == "1.3b":
        MODEL_PATH = "/home/ec2-user/model-server/models/deepseek-1.3b"
        quantize = False 
    else:
        MODEL_PATH = "/home/ec2-user/model-server/models/deepseek-6.7b"
        quantize = True  
    
    logger.info(f"Loading model from {MODEL_PATH}")
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if cuda_available:
        if quantize:
            # use 4bit quant
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            # load model with quant
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto"
            )
            logger.info("Model loaded successfully with 4-bit quantization on GPU")
        else:
            # load smaller model with half precision
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Model loaded successfully with FP16 on GPU")
    else:
        # fallback to cpu
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded successfully on CPU (fallback mode)")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    tokenizer = None

@app.route('/')
def home():
    if model is None:
        return "Model server is running but model failed to load. Check logs."
    
    if cuda_available:
        return "Health AI Model Server is running with GPU acceleration!"
    else:
        return "Health AI Model Server is running on CPU (slower performance)."

@app.route('/health')
def health_check():
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "cuda_available": cuda_available,
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else "None"
    }
    
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        logger.error("Prediction attempted but model is not loaded")
        return jsonify({"error": "Model not loaded properly. Check server logs."}), 500
        
    try:
        logger.info("Received prediction request")
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data received")
            return jsonify({"error": "No data provided"}), 400
            
        message = data.get('message', '')
        
        if not message:
            logger.warning("Empty message received")
            return jsonify({"error": "No message provided"}), 400
        
        logger.info(f"Processing message: {message}")
        
        # gen response
        encoded_input = tokenizer(
            message, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # move inputs to the same device as the model
        if cuda_available:
            encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                max_length=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated response of length {len(response)}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)