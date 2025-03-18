from flask import Flask, request, jsonify
import json
import logging
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# initialize flask app
app = Flask(__name__)

# try to detect if cuda is available
cuda_available = torch.cuda.is_available()
logger.info(f"cuda available: {cuda_available}")

if cuda_available:
    # output gpu info
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        logger.info(f"gpu {i}: {device_name}")
else:
    # set device to cpu if cuda not available
    device = 'cpu'
    logger.info(f"using device: {device}")

# Define the LSTM model architecture classes
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, coverage):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)) + coverage)
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        coverage += attention.unsqueeze(2)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention, coverage
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.attention = Attention(hidden_dim)

    def forward(self, x, hidden, cell, encoder_outputs, coverage):
        embedded = self.embedding(x)
        context, _, coverage = self.attention(hidden[-1], encoder_outputs, coverage)
        context = context.unsqueeze(1).repeat(1, embedded.size(1), 1)
        lstm_input = torch.cat((embedded, context), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(torch.cat((outputs, context), dim=2))
        return predictions, hidden, cell, coverage

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        coverage = torch.zeros_like(encoder_outputs)
        decoder_outputs, _, _, _ = self.decoder(decoder_inputs, hidden, cell, encoder_outputs, coverage)
        return decoder_outputs

# Register model classes as safe globals for PyTorch loading
torch.serialization.add_safe_globals([Seq2Seq, Encoder, Decoder, Attention])

# model and tokenizer globals
deepseek_model = None
deepseek_tokenizer = None
lstm_model = None
lstm_tokenizer = None

# load deepseek model
try:
    DEEPSEEK_MODEL_PATH = "/home/ec2-user/model-server/models/deepseek-7b"
    
    logger.info(f"loading deepseek model from {DEEPSEEK_MODEL_PATH}")
    
    # Create config.json if it doesn't exist
    config_path = os.path.join(DEEPSEEK_MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        logger.info("Creating config.json for DeepSeek model")
        config_data = {
            "model_type": "deepseek", 
            "architectures": ["DeepSeekForCausalLM"]
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    # load tokenizer
    deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH)
    
    if cuda_available:
        # use 4-bit quantization since this is a larger model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        # load model with quantization
        deepseek_model = AutoModelForCausalLM.from_pretrained(
            DEEPSEEK_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto"
        )
        logger.info("deepseek model loaded successfully with 4-bit quantization on gpu")
    else:
        # fallback to cpu (much slower)
        deepseek_model = AutoModelForCausalLM.from_pretrained(
            DEEPSEEK_MODEL_PATH, 
            low_cpu_mem_usage=True
        )
        logger.info("deepseek model loaded successfully on cpu (fallback mode)")
except Exception as e:
    logger.error(f"error loading deepseek model: {str(e)}")
    deepseek_model = None
    deepseek_tokenizer = None

# load lstm model
try:
    LSTM_MODEL_PATH = "/home/ec2-user/model-server/models/lstm/seq2seq_model.pth"
    
    logger.info(f"loading lstm model from {LSTM_MODEL_PATH}")
    
    # ensure distilbert tokenizer for lstm model
    lstm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
    lstm_tokenizer.add_special_tokens(special_tokens)
    
    # load lstm model - force using the safe loading by setting weights_only=False
    if cuda_available:
        lstm_model = torch.load(LSTM_MODEL_PATH, map_location=torch.device('cuda'), weights_only=False)
        logger.info("lstm model loaded successfully on gpu")
    else:
        lstm_model = torch.load(LSTM_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        logger.info("lstm model loaded successfully on cpu")
        
    # ensure model is in eval mode
    lstm_model.eval()
except Exception as e:
    logger.error(f"error loading lstm model: {str(e)}")
    lstm_model = None
    lstm_tokenizer = None

@app.route('/')
def home():
    status = []
    
    if deepseek_model is None:
        status.append("DeepSeek model failed to load")
    else:
        status.append("DeepSeek model loaded successfully")
        
    if lstm_model is None:
        status.append("LSTM model failed to load")
    else:
        status.append("LSTM model loaded successfully")
    
    if cuda_available:
        status.append("Server is running with GPU acceleration")
    else:
        status.append("Server is running on CPU (slower performance)")
    
    return "<br>".join(status)

@app.route('/health')
def health_check():
    status = {
        "status": "healthy" if (deepseek_model is not None or lstm_model is not None) else "unhealthy",
        "cuda_available": cuda_available,
        "deepseek_model_loaded": deepseek_model is not None,
        "deepseek_tokenizer_loaded": deepseek_tokenizer is not None,
        "lstm_model_loaded": lstm_model is not None,
        "lstm_tokenizer_loaded": lstm_tokenizer is not None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else "None"
    }
    
    return jsonify(status)

# helper function for lstm prediction
def predict_with_lstm(model, tokenizer, question, beam_width=5):
    model.eval()
    with torch.no_grad():
        input_text = f"question: {question}"
        
        # determine device
        device = next(model.parameters()).device
        
        # encode input using the tokenizer
        encodings = tokenizer(input_text, 
                            return_tensors='pt', 
                            padding='max_length',
                            truncation=True,
                            max_length=512)
        
        # move to appropriate device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        encoder_input = encodings['input_ids']
        
        # forward pass through encoder
        encoder_outputs, hidden, cell = model.encoder(encoder_input)
        coverage = torch.zeros_like(encoder_outputs)

        # beam search decoding (simplified version for inference)
        start_token_id = tokenizer.convert_tokens_to_ids("<start>")
        end_token_id = tokenizer.convert_tokens_to_ids("<end>")
        
        # initialize sequences with start token
        sequences = [([start_token_id], 0.0, hidden, cell, coverage)]
        
        # beam search parameters
        max_len = 150
        
        for _ in range(max_len):
            all_candidates = []
            
            for seq, score, hidden, cell, coverage in sequences:
                decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
                output, hidden, cell, coverage = model.decoder(decoder_input, hidden, cell, encoder_outputs, coverage)
                
                # get top-k tokens
                top_k_probs, top_k_indices = torch.topk(output[0, -1], beam_width)
                
                for i in range(len(top_k_indices)):
                    token = top_k_indices[i].item()
                    candidate_score = score + torch.log(top_k_probs[i]).item()
                    all_candidates.append((seq + [token], candidate_score, hidden, cell, coverage))
            
            # sort candidates and keep top beam_width
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # check if any sequence has generated the end token
            if any(end_token_id in seq for seq, _, _, _, _ in sequences):
                break
        
        # get the best sequence
        final_seq = sequences[0][0]
        
        # truncate sequence at end token if present
        if end_token_id in final_seq:
            final_seq = final_seq[:final_seq.index(end_token_id)]
        
        # remove start token if present
        if start_token_id in final_seq:
            final_seq = [token for token in final_seq if token != start_token_id]
        
        # decode tokens to text
        return tokenizer.decode(final_seq, skip_special_tokens=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("received prediction request")
        data = request.get_json()
        
        if not data:
            logger.warning("no json data received")
            return jsonify({"error": "no data provided"}), 400
            
        message = data.get('message', '')
        model_type = data.get('model_type', 'deepseek')  # default to deepseek if not specified
        
        if not message:
            logger.warning("empty message received")
            return jsonify({"error": "no message provided"}), 400
        
        logger.info(f"processing message with {model_type} model: {message}")
        
        # check which model to use
        if model_type == 'lstm':
            if lstm_model is None or lstm_tokenizer is None:
                logger.error("lstm model requested but not loaded")
                return jsonify({"error": "lstm model not loaded properly"}), 500
                
            # generate response using lstm model
            response = predict_with_lstm(lstm_model, lstm_tokenizer, message)
        else:  # default to deepseek
            if deepseek_model is None or deepseek_tokenizer is None:
                logger.error("deepseek model requested but not loaded")
                return jsonify({"error": "deepseek model not loaded properly"}), 500
                
            # generate response using deepseek model
            encoded_input = deepseek_tokenizer(
                message, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            # move inputs to the same device as the model
            if cuda_available:
                encoded_input = {k: v.to(deepseek_model.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = deepseek_model.generate(
                    input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=deepseek_tokenizer.eos_token_id
                )
            
            response = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"generated response of length {len(response)}")
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
[ec2-user@ip-172-31-69-62 model-server]$ 