from flask import Flask, request, jsonify
import json
import logging
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load only one model based on environment variable
MODEL_TO_LOAD = os.environ.get("MODEL_TO_LOAD", "lstm").lower()

# configure logging to file and console
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

# detect cuda availability and set device
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logger.info(f"cuda available: {cuda_available}")
logger.info(f"CONFIG: Only loading {MODEL_TO_LOAD} model as specified by environment variable")

# log gpu details if available
if cuda_available:
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        logger.info(f"gpu {i}: {device_name}")
else:
    logger.info("using device: cpu")

# lstm model definitions
class Attention(nn.Module):
    # initialize attention mechanism
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    # forward pass for attention
    def forward(self, hidden, encoder_outputs, coverage):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)) + coverage)
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        coverage += attention.unsqueeze(2)
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention, coverage

class Encoder(nn.Module):
    # initialize encoder
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    # forward pass for encoder
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    # initialize decoder
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.attention = Attention(hidden_dim)

    # forward pass for decoder
    def forward(self, x, hidden, cell, encoder_outputs, coverage):
        embedded = self.embedding(x)
        context, _, coverage = self.attention(hidden[-1], encoder_outputs, coverage)
        context = context.unsqueeze(1).repeat(1, embedded.size(1), 1)
        lstm_input = torch.cat((embedded, context), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(torch.cat((outputs, context), dim=2))
        return predictions, hidden, cell, coverage

class Seq2Seq(nn.Module):
    # initialize seq2seq model
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim)

    # forward pass for seq2seq
    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        coverage = torch.zeros_like(encoder_outputs)
        decoder_outputs, _, _, _ = self.decoder(decoder_inputs, hidden, cell, encoder_outputs, coverage)
        return decoder_outputs

# register safe globals for model loading
torch.serialization.add_safe_globals([Seq2Seq, Encoder, Decoder, Attention])

# globals for models
lstm_model = None
lstm_tokenizer = None
deepseek_model = None
deepseek_tokenizer = None
mediseek_model = None
mediseek_tokenizer = None

# load lstm model
if MODEL_TO_LOAD == "lstm":
    try:
        LSTM_MODEL_PATH = "/home/ec2-user/model-server/models/lstm/seq2seq_model.pth"
        logger.info(f"loading lstm model from {LSTM_MODEL_PATH}")
        lstm_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
        lstm_tokenizer.add_special_tokens(special_tokens)
        if cuda_available:
            lstm_model = torch.load(LSTM_MODEL_PATH, map_location=torch.device('cuda'), weights_only=False)
            logger.info("lstm model loaded successfully on gpu")
        else:
            lstm_model = torch.load(LSTM_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            logger.info("lstm model loaded successfully on cpu")
        lstm_model.eval()
        logger.info("lstm model set to evaluation mode")
    except Exception as e:
        logger.error(f"error loading lstm model: {str(e)}", exc_info=True)
        lstm_model = None
        lstm_tokenizer = None

# load base deepseek model
if MODEL_TO_LOAD == "deepseek":
    try:
        DEEPSEEK_MODEL_PATH = "/home/ec2-user/model-server/models/deepseek"
        logger.info(f"loading base deepseek model from {DEEPSEEK_MODEL_PATH}")
        config_path = os.path.join(DEEPSEEK_MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            logger.info("creating config.json for base deepseek model as llama type")
            config_data = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "_name_or_path": "deepseek-ai/deepseek-llm-7b-base"
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_PATH, trust_remote_code=True)
        if cuda_available:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            deepseek_model = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("base deepseek model loaded successfully with 4-bit quantization on gpu")
        else:
            deepseek_model = AutoModelForCausalLM.from_pretrained(
                DEEPSEEK_MODEL_PATH,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            logger.info("base deepseek model loaded successfully on cpu")
    except Exception as e:
        logger.error(f"error loading base deepseek model: {str(e)}", exc_info=True)
        deepseek_model = None
        deepseek_tokenizer = None

# load mediseek (fine-tuned) model
if MODEL_TO_LOAD == "mediseek":
    try:
        MEDISEEK_MODEL_PATH = "/home/ec2-user/model-server/models/mediseek"
        logger.info(f"loading mediseek model from {MEDISEEK_MODEL_PATH}")
        config_path = os.path.join(MEDISEEK_MODEL_PATH, "config.json")
        if not os.path.exists(config_path):
            logger.info("creating config.json for mediseek model as llama type")
            config_data = {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "_name_or_path": "mediseek-llm-7b"
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        mediseek_tokenizer = AutoTokenizer.from_pretrained(MEDISEEK_MODEL_PATH, trust_remote_code=True)
        fine_tuned_path = os.path.join(MEDISEEK_MODEL_PATH, "fine_tuned_deep_seek.pth")
        if os.path.exists(fine_tuned_path):
            logger.info("found fine_tuned_deep_seek.pth, loading state dict from it")
            state_dict = torch.load(fine_tuned_path, map_location="cuda" if cuda_available else "cpu")
            if cuda_available:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                mediseek_model = AutoModelForCausalLM.from_pretrained(
                    MEDISEEK_MODEL_PATH,
                    state_dict=state_dict,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("mediseek model loaded successfully with 4-bit quantization on gpu using fine_tuned_deep_seek.pth")
            else:
                mediseek_model = AutoModelForCausalLM.from_pretrained(
                    MEDISEEK_MODEL_PATH,
                    state_dict=state_dict,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("mediseek model loaded successfully on cpu using fine_tuned_deep_seek.pth")
        else:
            if cuda_available:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                mediseek_model = AutoModelForCausalLM.from_pretrained(
                    MEDISEEK_MODEL_PATH,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("mediseek model loaded successfully with 4-bit quantization on gpu (default loader)")
            else:
                mediseek_model = AutoModelForCausalLM.from_pretrained(
                    MEDISEEK_MODEL_PATH,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                logger.info("mediseek model loaded successfully on cpu (default loader)")
    except Exception as e:
        logger.error(f"error loading mediseek model: {str(e)}", exc_info=True)
        mediseek_model = None
        mediseek_tokenizer = None

# lstm inference function: beam search decoder
def beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width=3, max_len=150, length_penalty=0.8):
    # performs beam search with n-gram blocking and length penalty
    start_token_id = tokenizer.convert_tokens_to_ids("<start>")
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    sequences = [([start_token_id], 0.0, hidden, cell, coverage, set())]
    n = 2  # n-gram blocking variable

    # log start of beam search
    logger.debug("starting beam search decoding")
    for step in range(max_len):
        all_candidates = []
        for seq, score, hidden, cell, coverage, n_grams in sequences:
            decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
            output, hidden, cell, coverage = model.decoder(decoder_input, hidden, cell, encoder_outputs, coverage)
            top_k_probs, top_k_indices = torch.topk(output[0, -1], beam_width * 2)
            for i in range(len(top_k_indices)):
                token = top_k_indices[i].item()
                if len(seq) >= n - 1:
                    current_n_gram = tuple(seq[-(n-1):] + [token])
                    if current_n_gram in n_grams:
                        continue
                    new_n_grams = n_grams.copy()
                    new_n_grams.add(current_n_gram)
                else:
                    new_n_grams = n_grams.copy()
                candidate_score = score + torch.log(top_k_probs[i]).item() / (len(seq) ** length_penalty)
                all_candidates.append((seq + [token], candidate_score, hidden, cell, coverage, new_n_grams))
                if len(all_candidates) >= beam_width * 2:
                    break
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        if any(end_token_id in seq for seq, _, _, _, _, _ in sequences):
            logger.debug("end token found, stopping beam search")
            break
    final_seq = sequences[0][0]
    logger.info(f"generated token sequence length: {len(final_seq)}")
    if end_token_id in final_seq:
        final_seq = final_seq[:final_seq.index(end_token_id)]
    if start_token_id in final_seq:
        final_seq = [token for token in final_seq if token != start_token_id]
    decoded_response = tokenizer.decode(final_seq, skip_special_tokens=True)
    logger.debug(f"decoded response: {decoded_response[:100]}... (total length: {len(decoded_response)})")
    return decoded_response

# lstm inference function: predict answer
def predict_answer(model, tokenizer, question, beam_width=5):
    # generates output sequence using beam search
    model.eval()
    with torch.no_grad():
        input_text = f"question: {question}"
        encodings = tokenizer(
            input_text, 
            return_tensors='pt', 
            padding='max_length',
            truncation=True,
            max_length=512
        ).to(device)
        logger.debug(f"tokenized input length: {encodings['input_ids'].shape[1]}")
        encoder_input = encodings['input_ids']
        encoder_outputs, hidden, cell = model.encoder(encoder_input)
        coverage = torch.zeros_like(encoder_outputs).to(device)
        return beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width)

# flask endpoint: home
@app.route('/')
def home():
    # display model loading status
    status = []
    status.append(f"Single model server running {MODEL_TO_LOAD.upper()} model")
    status.append("deepseek model: " + ("loaded" if deepseek_model else "not loaded"))
    status.append("mediseek model: " + ("loaded" if mediseek_model else "not loaded"))
    status.append("lstm model: " + ("loaded" if lstm_model else "not loaded"))
    status.append("running on " + ("gpu" if cuda_available else "cpu (slower)"))
    logger.info("home endpoint accessed")
    return "<br>".join(status)

# flask endpoint: health check
@app.route('/health')
def health_check():
    # provide detailed health status
    status = {
        "model_type": MODEL_TO_LOAD,
        "status": "healthy" if (
            (MODEL_TO_LOAD == "lstm" and lstm_model is not None) or
            (MODEL_TO_LOAD == "deepseek" and deepseek_model is not None) or
            (MODEL_TO_LOAD == "mediseek" and mediseek_model is not None)
        ) else "unhealthy",
        "cuda_available": cuda_available,
        "deepseek_model_loaded": deepseek_model is not None,
        "deepseek_tokenizer_loaded": deepseek_tokenizer is not None,
        "mediseek_model_loaded": mediseek_model is not None,
        "mediseek_tokenizer_loaded": mediseek_tokenizer is not None,
        "lstm_model_loaded": lstm_model is not None,
        "lstm_tokenizer_loaded": lstm_tokenizer is not None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else "none"
    }
    logger.info("health check endpoint accessed")
    return jsonify(status)

# flask endpoint: predict
@app.route('/predict', methods=['POST'])
def predict():
    # handle prediction requests
    try:
        logger.info("received prediction request")
        data = request.get_json()
        if not data:
            logger.warning("no json data received")
            return jsonify({"error": "no data provided"}), 400

        message = data.get('message', '')
        model_type = data.get('model_type', MODEL_TO_LOAD)
        
        # Log if the requested model doesn't match what we loaded
        if model_type != MODEL_TO_LOAD:
            logger.warning(f"requested model type '{model_type}' doesn't match loaded model '{MODEL_TO_LOAD}', using {MODEL_TO_LOAD}")
            model_type = MODEL_TO_LOAD
            
        if not message:
            logger.warning("empty message received")
            return jsonify({"error": "no message provided"}), 400

        logger.info(f"processing message with {model_type} model: {message}")

        if model_type == 'lstm':
            if lstm_model is None or lstm_tokenizer is None:
                logger.error("lstm model requested but not loaded")
                return jsonify({"error": "lstm model not loaded properly"}), 500
            logger.debug("starting lstm inference")
            response = predict_answer(lstm_model, lstm_tokenizer, message, beam_width=5)
        elif model_type == 'mediseek':
            if mediseek_model is None or mediseek_tokenizer is None:
                logger.error("mediseek model requested but not loaded")
                return jsonify({"error": "mediseek model not loaded properly"}), 500
            logger.debug("starting mediseek inference")
            encoded_input = mediseek_tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            if cuda_available:
                encoded_input = {k: v.to(mediseek_model.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = mediseek_model.generate(
                    input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    max_length=512,
                    temperature=1.0,
                    top_p=0.7,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4,
                    do_sample=True,
                    pad_token_id=mediseek_tokenizer.eos_token_id
                )
            response = mediseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            if deepseek_model is None or deepseek_tokenizer is None:
                logger.error("deepseek model requested but not loaded")
                return jsonify({"error": "deepseek model not loaded properly"}), 500
            logger.debug("starting deepseek inference")
            encoded_input = deepseek_tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            if cuda_available:
                encoded_input = {k: v.to(deepseek_model.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                outputs = deepseek_model.generate(
                    input_ids=encoded_input['input_ids'],
                    attention_mask=encoded_input['attention_mask'],
                    max_length=512,
                    temperature=1.0,
                    top_p=0.7,
                    do_sample=True,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4,
                    pad_token_id=deepseek_tokenizer.eos_token_id
                )
            response = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Add model signature to the response
        model_signature = f"[Generated by {model_type.upper()} model]"
        response_with_signature = f"{response}\n\n{model_signature}"

        logger.info(f"generated response of length {len(response)} with {model_type} model")
        logger.debug(f"response preview: {response[:100]}...")
        return jsonify({"response": response_with_signature})
    except Exception as e:
        logger.error(f"error processing request with model {model_type if 'model_type' in locals() else MODEL_TO_LOAD}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# error handler for 404
@app.errorhandler(404)
def not_found(e):
    logger.warning("404 error: endpoint not found")
    return jsonify({"error": "endpoint not found"}), 404

# error handler for 500
@app.errorhandler(500)
def server_error(e):
    logger.error("500 error: internal server error", exc_info=True)
    return jsonify({"error": "internal server error"}), 500

# run the app
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    logger.info(f"starting flask app on host 0.0.0.0, port {args.port}")
    app.run(host='0.0.0.0', port=args.port)  