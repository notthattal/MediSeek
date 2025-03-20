import torch
import json
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import BERTScorer
import nltk
from tqdm import tqdm
from traditional_model.traditional_model import Seq2Seq, Encoder, Decoder, Attention, beam_search_decoder
# Download required NLTK packages
nltk.download('wordnet')
nltk.download('punkt')

def load_test_data(folder_path, test_split=0.1, random_seed=42):
    """
    Load test data from JSON files
    """
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if not data['qa_pairs']:
                continue
                
            qa_pairs = data['qa_pairs']
            df = pd.DataFrame(qa_pairs)
            all_data.append(df)
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Split into train and test sets
    np.random.seed(random_seed)
    mask = np.random.rand(len(combined_df)) < test_split
    test_df = combined_df[mask]
    
    return test_df

def evaluate_model(model, tokenizer, test_data, device='cpu', beam_width=5):
    """
    Evaluate model using various metrics
    """
    model.eval()
    
    # Initialize metric calculators
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(rouge_metrics, use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    smooth = SmoothingFunction().method1
    
    results = {
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': [],
        'bleu': [],
        'meteor': [],
        'bertscore_f1': [],
        'predictions': [],
        'references': [],
        'questions': []
    }
    
    # Define predict_answer function if not available directly from model
    def predict_answer_from_model(question):
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

            # Use beam search decoding from the original script
            from traditional_model import beam_search_decoder
            return beam_search_decoder(model, tokenizer, encoder_outputs, hidden, cell, coverage, beam_width)
    
    # Process each question-answer pair
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model"):
        question = row['question']
        reference = row['answer']
        
        # Generate prediction
        prediction = predict_answer_from_model(question)
        
        # Calculate ROUGE scores
        rouge_scores = scorer.score(reference, prediction)
        results['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
        results['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
        results['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
        
        # Calculate BLEU score
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        bleu = corpus_bleu([reference_tokens], [prediction_tokens], smoothing_function=smooth)
        results['bleu'].append(bleu)
        
        # Calculate METEOR score
        meteor = meteor_score([reference_tokens[0]], prediction_tokens)
        results['meteor'].append(meteor)
        
        # Store the actual texts for BERTScore batch processing and reference
        results['predictions'].append(prediction)
        results['references'].append(reference)
        results['questions'].append(question)
    
    # Calculate BERTScore (batch processing for efficiency)
    P, R, F1 = bert_scorer.score(results['predictions'], results['references'])
    results['bertscore_f1'] = F1.tolist()
    
    # Calculate average metrics
    avg_metrics = {
        'avg_rouge1_f': np.mean(results['rouge1_f']),
        'avg_rouge2_f': np.mean(results['rouge2_f']),
        'avg_rougeL_f': np.mean(results['rougeL_f']),
        'avg_bleu': np.mean(results['bleu']),
        'avg_meteor': np.mean(results['meteor']),
        'avg_bertscore_f1': np.mean(results['bertscore_f1'])
    }
    
    return results, avg_metrics

def save_evaluation_results(results, avg_metrics, output_file="evaluation_results.json"):
    """
    Save evaluation results to a file
    """
    output = {
        'average_metrics': avg_metrics,
        'detailed_results': []
    }
    
    # Compile detailed results
    for i in range(len(results['questions'])):
        output['detailed_results'].append({
            'question': results['questions'][i],
            'reference': results['references'][i],
            'prediction': results['predictions'][i],
            'rouge1_f': results['rouge1_f'][i],
            'rouge2_f': results['rouge2_f'][i],
            'rougeL_f': results['rougeL_f'][i],
            'bleu': results['bleu'][i],
            'meteor': results['meteor'][i],
            'bertscore_f1': results['bertscore_f1'][i]
        })
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Evaluation results saved to {output_file}")

def visualize_results(results, avg_metrics, output_file="evaluation_visualization.html"):
    """
    Create a visualization of evaluation results
    """
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    
    # Create figures
    plt.figure(figsize=(10, 6))
    metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'meteor', 'bertscore_f1']
    avg_values = [avg_metrics[f'avg_{m}'] for m in metrics]
    
    plt.bar(metrics, avg_values)
    plt.title('Average Performance Across Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Save to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        axes[i].hist(results[metric], bins=20)
        axes[i].set_title(f'{metric} Distribution')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    img_str2 = base64.b64encode(buffer2.read()).decode()
    plt.close()
    
    # Create an HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metrics {{ margin-bottom: 30px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .sample {{ margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Results</h1>
        
        <div class="metrics">
            <h2>Average Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
    """
    
    # Add average metrics to the table
    for metric, value in avg_metrics.items():
        html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{value:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="visualizations">
            <h2>Visualizations</h2>
            <img src="data:image/png;base64,{}" alt="Average Metrics" width="800">
            <img src="data:image/png;base64,{}" alt="Metric Distributions" width="1000">
        </div>
        
        <div class="sample">
            <h2>Sample Predictions (10 Examples)</h2>
            <table>
                <tr>
                    <th>Question</th>
                    <th>Reference</th>
                    <th>Prediction</th>
                    <th>ROUGE-L</th>
                    <th>BERTScore</th>
                </tr>
    """.format(img_str, img_str2)
    
    # Add sample predictions
    for i in range(min(10, len(results['questions']))):
        html_content += f"""
                <tr>
                    <td>{results['questions'][i]}</td>
                    <td>{results['references'][i]}</td>
                    <td>{results['predictions'][i]}</td>
                    <td>{results['rougeL_f'][i]:.4f}</td>
                    <td>{results['bertscore_f1'][i]:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_file}")

def main():
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_data = load_test_data("./data/qa_pairs/huberman_lab", test_split=0.1)
    print(f"Loaded {len(test_data)} test examples")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    vocab_size = len(tokenizer)
    embedding_dim = 256
    hidden_dim = 256
    batch_size = 32

    # Create model
    model = Seq2Seq(vocab_size, embedding_dim, hidden_dim).to(device)
    checkpoint = torch.load('seq2seq_model.pth', map_location=device, weights_only=False)
    model = checkpoint
    model.to(device)
    
    # Evaluate model
    print("Evaluating model...")
    results, avg_metrics = evaluate_model(model, tokenizer, test_data, device)
    
    # # Print average metrics
    print("\nAverage Evaluation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # # Save results
    save_evaluation_results(results, avg_metrics)
    
    # Visualize results
    try:
        visualize_results(results, avg_metrics)
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    main()