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
    '''
    Load The Question-Answer Pairs as a pandas Dataframe

    Input:
        - file_path (str): The path to the question-answer pairs
    
    Returns:
        - df (pd.DataFrame): The question-answer pairs as a pandas dataframe
    '''
    # load the specified json file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # if the file does not contain qa_pairs return None
    if not data['qa_pairs']:
        return None
    
    # load the QA pairs
    qa_pairs = data['qa_pairs']

    # create the dataframe
    df = pd.DataFrame(qa_pairs)

    return df

def create_features(df):
    '''
    Prepares the input text and target text
    
    Input:
        - df (pd.DataFrame) - The QA dataframe
    
    Returns:
        - The input and target columns
    '''
    # specify that the input is a question
    df['input_text'] = "question: " + df['question']

    # add start and stop tokens to the answer
    df['target_text'] = "<start> " + df['answer'] + " <end>"

    return df[['input_text', 'target_text']]

def clean_text(text):
    '''
    Cleans the text data

    Inputs:
        - text (str): The text to be cleaned
    
    Returns:
        - the cleaned text
    '''

    # remove leading/trailing spaces, puts all text on a single line without tabs and converts it to lowercase
    return text.strip().replace('\n', ' ').replace('\t', ' ').lower()

def clean_data(df):
    '''
    Removes leading/trailing spaces, puts all text on a single line without tabs and converts it to lowercase for all input and target
    text for the dataframe

    Input:
        - df (pd.DataFrame) - The dataframe to be cleaned
    
    Returns:
        - the cleaned dataframe
    '''

    # clean text for inputs
    df['input_text'] = df['input_text'].apply(clean_text)

    # clean text for targets
    df['target_text'] = df['target_text'].apply(clean_text)
    
    return df

def load_all_data(folder_path):
    '''
    Creates a dataframe for all json files containing Question-Answer Pairs within a specified folder

    Input:
        - folder_path (str): The folder path to create the dataframe from
    
    Returns:
        - combined_df (pd.DataFrame): The dataframe of all Question and answer pairs
    '''
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            # get the proper file path
            file_path = os.path.join(folder_path, file_name)

            # load the QA pairs as a dataframe
            df = load_data(file_path)
            
            # skip if that json file does not contain QA pairs
            if df is None:
                continue

            # create the Input and Target Columns with proper formatting
            df = create_features(df)
            df = clean_data(df)

            # add to the list of dataframes to be concatenated
            all_data.append(df)
    
    # concatenate all dataframes together to have one final DF
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

'''
Dataset Class for the Sequence-to-Sequence Model

Class Members:
    - encoder_inputs: the inputs to the encoder
    - decoder_inputs: the inputs to the decoder
    - decoder_outputs: the ground truth output of the decoder should expect (i.e. the targets)
'''
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

'''
The attention Mechanism to be used in the decoder

Class Members:
    - attn: The linear layer used to compute the energy
    - v: The linear layer used to help calculate the final attention scores
'''
# The below class code was generated using Claude 3.7 Sonnet on 3/14/25 at 7:30 p.m.
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, coverage):
        # get the sequence length
        seq_len = encoder_outputs.shape[1]

        # change shape of hidden dimension to [batch_size, 1, hidden_dim]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # combine coverage vector with attention and computes the hyperbolic tangent
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)) + coverage)

        # runs the energy through a linear layer and computes the softmax attention scores
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)

        # update the coverage vector
        coverage += attention.unsqueeze(2)

        # perform batch matrix multiplication to compute the weighted sum of encoder outputs 
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention, coverage

'''
The Encoder LSTM Class

Class Members:
    - embedding: The embedding layer
    - lstm: The LSTM layer
'''    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        # create the embedding of the input
        embedded = self.embedding(x)

        # pass the embedding through the lstm and retrieve the outputs
        outputs, (hidden, cell) = self.lstm(embedded)

        return outputs, hidden, cell

'''
The Encoder LSTM Class

Class Members:
    - embedding: The embedding layer
    - lstm: The LSTM layer
    - fc: The fully connected layer
    - attention: The attention mechanism
'''    
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        self.attention = Attention(hidden_dim) # This code snippet was generated using Claude 3.7 Sonnet on 3/14/25 at 7:30 p.m.

    # The below function was modified using Claude 3.7 Sonnet on 3/14/25 at 7:30 p.m.
    def forward(self, x, hidden, cell, encoder_outputs, coverage):
        # create the embeddings of the input
        embedded = self.embedding(x)

        # perform attention over the most recent hidden state and encoder outputs
        context, _, coverage = self.attention(hidden[-1], encoder_outputs, coverage)
        context = context.unsqueeze(1).repeat(1, embedded.size(1), 1)

        # concatenate embeddings with context vector 
        lstm_input = torch.cat((embedded, context), dim=2)

        # get outputs from the lstm
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # run outputs and context through a final fully connected layer to retrieve predictions
        predictions = self.fc(torch.cat((outputs, context), dim=2))

        return predictions, hidden, cell, coverage

'''
The Full Sequence-to-Sequence Model

Class Members:
    - encoder: The encoder module
    - decoder: The decoder module
'''  
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

    def forward(self, encoder_inputs, decoder_inputs):
        # pass encoder_inputs to the encoder
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)

        # initialize the coverage vector for attention
        coverage = torch.zeros_like(encoder_outputs).to(device) # This code snippet was generated using Claude 3.7 Sonnet on 3/14/25 at 7:30 p.m.

        # retrieve predictions from the decoder
        decoder_outputs, _, _, _ = self.decoder(decoder_inputs, hidden, cell, encoder_outputs, coverage)

        return decoder_outputs

def beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width=3, max_len=150, length_penalty=0.8):
    '''
    Performs a beam search with n-gram blocking and uses a length penalty to mitigate generating incomplete sequences

    Inputs:
        - model: the fully trained sequence-to-sequence model
        - tokenizer: the same tokenizer used for training
        - encoder_outputs: the outputs of the encoder
        - hidden: the hidden state output from the encoder
        - cell: the cell state output from the encoder
        - coverage: the coverage vector
        - beam_width: the length of the beam search
        - max_len: the max length of the output generated sequence
        - length_penalty: the length penalty factor
    
    Returns:
        - The best sequence predicted by the trained model 
    '''
    # Get token IDs for special tokens
    start_token_id = tokenizer.convert_tokens_to_ids("<start>")
    end_token_id = tokenizer.convert_tokens_to_ids("<end>")
    
    # Initialize sequences with start token and add an empty n-gram set to each sequence
    sequences = [([start_token_id], 0.0, hidden, cell, coverage, set())]
    
    # n-gram blocking variable
    n = 2

    for _ in range(max_len):
        all_candidates = [] 

        for seq, score, hidden, cell, coverage, n_grams in sequences:
            # use the last output as the new input
            decoder_input = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)

            # get decoder's prediction for the current sequence
            output, hidden, cell, coverage = model.decoder(decoder_input, hidden, cell, encoder_outputs, coverage)

            # get top-k tokens
            top_k_probs, top_k_indices = torch.topk(output[0, -1], beam_width * 2)

            for i in range(len(top_k_indices)):
                token = top_k_indices[i].item()
                
                # if the sequence >= n-blocking variable check for n-grams
                if len(seq) >= n-1:
                    # get current gram
                    current_n_gram = tuple(seq[-(n-1):] + [token])
                    # skip this token if the n-gram is already in this sequence's set
                    if current_n_gram in n_grams:
                        continue
                    # create a new n-gram set for this candidate by copying the current one
                    new_n_grams = n_grams.copy()
                    new_n_grams.add(current_n_gram)
                else:
                    new_n_grams = n_grams.copy()
                
                # get the candidate score by getting the log of the probability of this token over the length of the sequence, penalizing the model for generating sequences which are too short
                candidate_score = score + torch.log(top_k_probs[i]).item() / (len(seq) ** length_penalty)
                
                # add possible candidates to the final list
                all_candidates.append((seq + [token], candidate_score, hidden, cell, coverage, new_n_grams))
                
                # break if we have enough candidates
                if len(all_candidates) >= beam_width * 2:
                    break

        # sort candidates and keep top beam_width
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # check if any sequence has generated the end token
        if any(end_token_id in seq for seq, _, _, _, _, _ in sequences):
            break

    # get the best sequence
    final_seq = sequences[0][0]
    
    # truncate sequence at end token if present
    if end_token_id in final_seq:
        final_seq = final_seq[:final_seq.index(end_token_id)]
    
    # remove start token if present
    if start_token_id in final_seq:
        final_seq = [token for token in final_seq if token != start_token_id]
    
    # decode tokens to text using the tokenizer's built-in decoder
    return tokenizer.decode(final_seq, skip_special_tokens=True)

def train_model(model, dataloader, vocab_size, criterion, optimizer, num_epochs=10):
    '''
    Training function for the Seq2Seq Model

    Inputs:
        - model: the untrained Seq2Seq model
        - dataloader: the dataloader to be used for training
        - vocab_size: the size of our vocabulary
        - criterion: the criterion to train the model on
        - optimizer: the optimizer to be used for training
        - num_epochs: the number of epochs to train the model for
    '''
    # set model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # run forward pass on the model
            outputs = model(batch['encoder_inputs'], batch['decoder_inputs'])
            
            # reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, vocab_size)
            targets = batch['decoder_outputs'].view(-1)
            
            # calculate the loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # backward pass and optimizer step
            loss.backward()
            optimizer.step()
        
        # print loss metrics
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

def predict_answer(model, tokenizer, question, beam_width=5):
    '''
    The function to be used at inference for performing beam search and generating the output sequence

    Inputs:
        - model: The trained Seq2Seq model
        - tokenizer: The same tokenizer used to prep data for training
        - quesiton: The user query
        - beam_width: The beam_width we will use for beam search
    '''
    # set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # format the query similar to training
        input_text = f"question: {question}"
        
        # encode input_text using the tokenizer
        encodings = tokenizer(input_text, 
                              return_tensors='pt', 
                              padding='max_length',
                              truncation=True,
                              max_length=512).to(device)
        
        # get input to be fed to the model's encoder
        encoder_input = encodings['input_ids']
        
        # forward pass through encoder
        encoder_outputs, hidden, cell = model.encoder(encoder_input)

        # initialize the coverage vector
        coverage = torch.zeros_like(encoder_outputs).to(device)

        # perform beam search decoding
        return beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width)

def main():
    # set QA-pair folder
    folder_path = "../../data/qa_pairs/huberman_lab"

    # get the dataframe of all QA pairs
    df = load_all_data(folder_path)

    # initialize the tokenizer to be used
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # set the start and end tokens
    special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
    tokenizer.add_special_tokens(special_tokens)

    # convert text to sequences
    X = tokenizer(df['input_text'].tolist(), padding='max_length', max_length=512, truncation=True, return_tensors='pt')['input_ids']
                
    y_texts = df['target_text'].tolist()
    y = tokenizer(y_texts, padding='max_length', max_length=150, truncation=True, return_tensors='pt')['input_ids']

    # prepare decoder input/output data
    decoder_input_data = torch.roll(y, shifts=1, dims=1)
    start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])
    decoder_input_data[:, 0] = start_token_id
    decoder_output_data = y

    # initialize model parameters
    vocab_size = len(tokenizer)
    embedding_dim = 256
    hidden_dim = 256
    batch_size = 32

    # create the dataset and dataloader
    dataset = QADataset(X, decoder_input_data, decoder_output_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # instantiate the model
    model = Seq2Seq(vocab_size, embedding_dim, hidden_dim).to(device)

    # set the loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters())

    # perform model training
    train_model(model, dataloader, vocab_size, criterion, optimizer, num_epochs=50)

    # print an example query response
    print(predict_answer(model, tokenizer, "How can omega-3 fatty acids benefit brain health?", beam_width=5))

    # save the model
    torch.save(model, '../../models/seq2seq_model.pth')

if __name__ == '__main__':
    main()