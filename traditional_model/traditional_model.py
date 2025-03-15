import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    if not data['qa_pairs']:
        return None
    
    qa_pairs = data['qa_pairs']
    df = pd.DataFrame(qa_pairs)
    return df

def create_features(df):
    df['input_text'] = "question: " + df['question']
    df['target_text'] = "<start> " + df['answer'] + " <end>"
    return df[['input_text', 'target_text']]

def clean_text(text):
    return text.strip().replace('\n', ' ').replace('\t', ' ').lower()

def clean_data(df):
    df['input_text'] = df['input_text'].apply(clean_text)
    df['target_text'] = df['target_text'].apply(clean_text)
    return df

def load_all_data(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            df = load_data(file_path)
            
            if df is None:
                continue

            df = create_features(df)
            df = clean_data(df)
            all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

class QADataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_outputs):
        self.encoder_inputs = torch.LongTensor(encoder_inputs).to(device)
        self.decoder_inputs = torch.LongTensor(decoder_inputs).to(device)
        self.decoder_outputs = torch.LongTensor(decoder_outputs).to(device)
    
    def __len__(self):
        return len(self.encoder_inputs)
    
    def __getitem__(self, idx):
        return {
            'encoder_inputs': self.encoder_inputs[idx],
            'decoder_inputs': self.decoder_inputs[idx],
            'decoder_outputs': self.decoder_outputs[idx]
        }

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, coverage):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Combine coverage vector with attention
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)) + coverage)
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)

        # Track cumulative attention (coverage)
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
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        coverage = torch.zeros_like(encoder_outputs).to(device)
        decoder_outputs, _, _, _ = self.decoder(decoder_inputs, hidden, cell, encoder_outputs, coverage)
        return decoder_outputs

def beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width=3, max_len=150, length_penalty=0.8):
    # Get token IDs for special tokens
    start_token_id = tokenizer.convert_tokens_to_ids("<start>")
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    
    # Initialize sequences with start token
    sequences = [([start_token_id], 0.0, hidden, cell, coverage, set())]  # Add empty n-gram set to each sequence
    
    n = 2  # Block 2-grams

    for _ in range(max_len):
        all_candidates = [] 

        for seq, score, hidden, cell, coverage, n_grams in sequences:
            decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
            output, hidden, cell, coverage = model.decoder(decoder_input, hidden, cell, encoder_outputs, coverage)

            # Get top-k tokens
            top_k_probs, top_k_indices = torch.topk(output[0, -1], beam_width * 2)  # Get more candidates to allow for filtering

            for i in range(len(top_k_indices)):
                token = top_k_indices[i].item()
                
                # Only check n-grams if we have enough tokens
                if len(seq) >= n-1:
                    current_n_gram = tuple(seq[-(n-1):] + [token])
                    # Skip this token if the n-gram is already in this sequence's set
                    if current_n_gram in n_grams:
                        continue
                    # Create a new n-gram set for this candidate by copying the current one
                    new_n_grams = n_grams.copy()
                    new_n_grams.add(current_n_gram)
                else:
                    new_n_grams = n_grams.copy()
                
                candidate_score = score + torch.log(top_k_probs[i]).item() / (len(seq) ** length_penalty)
                all_candidates.append((seq + [token], candidate_score, hidden, cell, coverage, new_n_grams))
                
                # Break if we have enough candidates
                if len(all_candidates) >= beam_width * 2:
                    break

        # Sort candidates and keep top beam_width
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Check if any sequence has generated the end token
        if any(end_token_id in seq for seq, _, _, _, _, _ in sequences):
            break

    # Get the best sequence
    final_seq = sequences[0][0]
    
    # Truncate sequence at end token if present
    if end_token_id in final_seq:
        final_seq = final_seq[:final_seq.index(end_token_id)]
    
    # Remove start token if present
    if start_token_id in final_seq:
        final_seq = [token for token in final_seq if token != start_token_id]
    
    # Decode tokens to text using the tokenizer's built-in decoder
    return tokenizer.decode(final_seq, skip_special_tokens=True)

def train_model(model, dataloader, vocab_size, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch['encoder_inputs'], batch['decoder_inputs'])
            
            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = batch['decoder_outputs'].view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def predict_answer(model, tokenizer, question, beam_width=5):
    model.eval()
    with torch.no_grad():
        input_text = f"question: {question}"
        
        # Encode input using the tokenizer
        encodings = tokenizer(input_text, 
                              return_tensors='pt', 
                              padding='max_length',
                              truncation=True,
                              max_length=512).to(device)
        
        encoder_input = encodings['input_ids']
        
        # Forward pass through encoder
        encoder_outputs, hidden, cell = model.encoder(encoder_input)
        coverage = torch.zeros_like(encoder_outputs).to(device)

        # Beam search decoding
        return beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width)

def main():
    folder_path = "./data/qa_pairs/huberman_lab"
    df = load_all_data(folder_path)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
    tokenizer.add_special_tokens(special_tokens)

    # Convert text to sequences
    X = tokenizer(df['input_text'].tolist(), padding='max_length', max_length=512, truncation=True, return_tensors='pt')['input_ids']
                
    y_texts = df['target_text'].tolist()
    y = tokenizer(y_texts, padding='max_length', max_length=150, truncation=True, return_tensors='pt')['input_ids']

    # Prepare decoder input/output data
    decoder_input_data = torch.roll(y, shifts=1, dims=1)
    start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])
    decoder_input_data[:, 0] = start_token_id
    decoder_output_data = y

    # Model parameters
    vocab_size = len(tokenizer)
    embedding_dim = 256
    hidden_dim = 256
    batch_size = 32

    # Create dataset and dataloader
    dataset = QADataset(X, decoder_input_data, decoder_output_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = Seq2Seq(vocab_size, embedding_dim, hidden_dim).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters())

    train_model(model, dataloader, vocab_size, criterion, optimizer, num_epochs=50)

    print(predict_answer(model, tokenizer, "How can omega-3 fatty acids benefit brain health?", beam_width=5))

    torch.save(model, 'seq2seq_model.pth')

if __name__ == '__main__':
    main()