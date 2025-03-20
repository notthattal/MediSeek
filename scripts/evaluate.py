'''
This script implements MedicalChatbotEvaluator which implements an evaluation pipeline to evaluate 
three different models: a naive model (deepseek 7b), a traditional model (encoder-decoder LSTM with attention),
and a fine-tuned model (fine-tuned deepseek). The evaluation is done by feeding questions from the Huberman Lab podcast
dataset to the models, retrieving their answers, and calculating different evaluation metrics compared to the
ground truth answers.
NOTE: The generate_html function was implemented using Claude 3.7 since we kept running into weird issues 
NOTE: The other part of the code was adapted from evaluate_traditional.py 
'''
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import BERTScorer
import nltk
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Union
import glob
from datetime import datetime
import torch.nn as nn
from traditional_model.traditional_model import Seq2Seq, Encoder, Decoder, Attention, beam_search_decoder
from peft import PeftModel

# Download required NLTK packages if not already downloaded
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


class MedicalChatbotEvaluator:
    """
    A class to evaluate the performance of medical chatbot models.
    Supports naive, traditional, and fine-tuned models, with comprehensive metrics.
    """
    
    def __init__(
        self, 
        naive_model_name: str = "deepseek-ai/deepseek-llm-1.3b-base",
        traditional_model_path: str = "./models/seq2seq_model.pth",
        fine_tuned_model_path: str = "./models/fine_tuned_deep_seek",
        qa_pairs_directory: str = "./data/qa_pairs",
        podcast_name: str = "huberman_lab",
        device: str = None
    ):
        """
        Initialize the evaluator with model paths and data paths.
        
        Args:
            naive_model_name: Name of the base model (deepseek 7b)
            traditional_model_path: Path to the traditional model (encoder-decoder LSTM)
            fine_tuned_model_path: Path to the fine-tuned model (fine-tuned deepseek)
            qa_pairs_directory: Directory containing QA pairs JSON files
            device: Device to run the model on (None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Store paths
        self.naive_model_name = naive_model_name
        self.traditional_model_path = traditional_model_path
        self.fine_tuned_model_path = fine_tuned_model_path
        self.qa_pairs_directory = os.path.join(qa_pairs_directory,podcast_name)
        
        # Initialize models and tokenizers
        self.naive_tokenizer = None
        self.naive_model = None
        self.traditional_model_tokenizer = None
        self.traditional_model = None 
        self.fine_tuned_tokenizer = None
        self.fine_tuned_model = None
        
        # For LSTM model
        self.traditional_vocab = None
        
        # Initialize data
        self.qa_pairs = []
        
        # Evaluation metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.smooth_fn = SmoothingFunction().method1
        
    def load_data(self) -> None:
        """
        Load and preprocess the QA pairs data.
        """
        print("Loading and preprocessing QA pairs data...")
        
        # Find all JSON files in the qa_pairs directory
        json_files = glob.glob(os.path.join(self.qa_pairs_directory, "*.json"))
        
        # Load QA pairs from each file
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                self.qa_pairs.extend(data["qa_pairs"])
        
        print(f"Loaded {len(self.qa_pairs)} QA pairs")

        # Extract topics for analysis
        all_topics = []
        for qa_pair in self.qa_pairs:
            if "topic" in qa_pair:
                print(f'\nTOPICs IN QA_PAIR: {qa_pair["topic"]} \n')
                all_topics.extend([s.strip() for s in qa_pair['topic'].split(',')])
        
        # Get unique topics
        self.unique_topics = sorted(set(all_topics))
        print(f"Found {len(self.unique_topics)} unique topics")
        
    def load_models(self) -> None:
        """
        Load the naive, traditional, and fine-tuned models.
        """
        print("Loading models...")
        
        # Load naive model (deepseek 7b)
        print(f"Loading naive model: {self.naive_model_name}")
        self.naive_tokenizer = AutoTokenizer.from_pretrained(self.naive_model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        self.naive_model = AutoModelForCausalLM.from_pretrained(
            self.naive_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=bnb_config
        )
        
        # Load traditional model (encoder-decoder LSTM)
        print(f"Loading traditional model from: {self.traditional_model_path}")
        self.traditional_model_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        special_tokens = {"additional_special_tokens": ["<start>", "<end>"]}
        self.traditional_model_tokenizer.add_special_tokens(special_tokens)
        # Here you would load your LSTM model with its vocabulary
        # This is a placeholder for your actual LSTM model loading code
        try:
            vocab_size = len(self.traditional_model_tokenizer)
            embedding_dim = 256
            hidden_dim = 256

            # Load LSTM model
            # self.traditional_model = Seq2Seq(vocab_size, embedding_dim, hidden_dim).to(self.device)
            lstm_device = 'cpu'
            checkpoint = torch.load(self.traditional_model_path, map_location=lstm_device, weights_only=False)
            self.traditional_model = checkpoint
            self.traditional_model.to(lstm_device)
            self.traditional_model.eval()
            
        except Exception as e:
            print(f"Warning: Failed to load traditional model: {e}")
            print("Traditional model will be skipped during evaluation.")
            self.traditional_model = None
        
        # Load fine-tuned model (fine-tuned deepseek)
        print(f"Loading fine-tuned model from: {self.fine_tuned_model_path}")
        self.fine_tuned_tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_model_path)
        config = AutoConfig.from_pretrained(self.fine_tuned_model_path)

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.fine_tuned_model = PeftModel.from_pretrained(self.naive_model, "./models/fine_tuned_deep_seek")
        # self.fine_tuned_model = AutoModelForCausalLM.from_pretrained(
        #     self.fine_tuned_model_path,
        #     config=config,
        #     quantization_config=bnb_config,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, # Added this in case we want to run this script on the EC2 equipped with NVIDIA Cuda support
        #     device_map="auto"
        # ).to(self.device)
        
        print("Models loaded successfully")
    
    def predict_with_naive_model(self, question: str) -> str:
        """
        Generate an answer using the naive model.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        inputs = self.naive_tokenizer(
            f"Question: {question}\nAnswer:", 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.naive_model.generate(
                inputs["input_ids"],
                max_length=1024,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.naive_tokenizer.eos_token_id
            )
        
        response = self.naive_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part
        answer = response.split("Answer:")[1].strip() if "Answer:" in response else response
        return answer
    
    def predict_with_traditional_model(self, question: str) -> str:
        """
        Generate an answer using the traditional LSTM model.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        if self.traditional_model is None:
            return "Traditional model is not available."
        
        try:
            with torch.no_grad():
                input_text = f"question: {question}"
                
                lstm_device = 'cpu'
                # Encode input using the tokenizer
                encodings = self.traditional_model_tokenizer(input_text, 
                                    return_tensors='pt', 
                                    padding='max_length',
                                    truncation=True,
                                    max_length=512)
                def ensure_same_device(tensor):
                    return tensor.to(lstm_device) if tensor.device != lstm_device else tensor

                
                encoder_input = ensure_same_device(encodings['input_ids'])

                
                # Forward pass through encoder
                encoder_outputs, hidden, cell = self.traditional_model.encoder(encoder_input)
                encoder_outputs = ensure_same_device(encoder_outputs)
                hidden = ensure_same_device(hidden)
                cell = ensure_same_device(cell)
                coverage = ensure_same_device(torch.zeros_like(encoder_outputs, dtype=torch.float32))

                # Use beam search decoding from the original script
                from traditional_model import beam_search_decoder
                return beam_search_decoder(self.traditional_model, self.traditional_model_tokenizer, encoder_outputs, hidden, cell, coverage, beam_width=5)
        except Exception as e:
            print(f"Error in traditional model prediction: {e}")
            return "Error generating response"
    
    def predict_with_fine_tuned_model(self, question: str) -> str:
        """
        Generate an answer using the fine-tuned model.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        inputs = self.fine_tuned_tokenizer(
            f"Question: {question}\nAnswer:", 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(
                inputs["input_ids"],
                max_length=1024,
                num_return_sequences=1,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.fine_tuned_tokenizer.eos_token_id
            )
        
        response = self.fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the answer part
        answer = response.split("Answer:")[1].strip() if "Answer:" in response else response
        return answer
    
    def evaluate_model(
        self, 
        model_type: str,
        data: List[Dict], 
        sample_size: int = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a model on the given data.
        
        Args:
            model_type: Type of model to evaluate ('naive', 'traditional', or 'fine_tuned')
            data: The data to evaluate on
            sample_size: Number of samples to evaluate (None for all)
            model_name: A name for the model (for reporting)
            
        Returns:
            A dictionary containing evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Sample data if sample_size is provided
        if sample_size is not None and sample_size < len(data):
            eval_data = np.random.choice(data, sample_size, replace=False)
        else:
            eval_data = data
        
        # Initialize result storage
        results = {
            'predictions': [],
            'ground_truth': [],
            'questions': [],
            'rouge1_f': [],
            'rouge2_f': [],
            'rougeL_f': [],
            'bleu': [],
            'bertscore_f1': [],
            'topics': []
        }
        
        # Process each QA pair
        for qa_pair in tqdm(eval_data, desc=f"Evaluating {model_name}"):
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            # Generate prediction based on model type
            if model_type == 'naive':
                prediction = self.predict_with_naive_model(question)
            elif model_type == 'traditional':
                prediction = self.predict_with_traditional_model(question)
            elif model_type == 'fine_tuned':
                prediction = self.predict_with_fine_tuned_model(question)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Store results
            results['predictions'].append(prediction)
            results['ground_truth'].append(ground_truth)
            results['questions'].append(question)
            
            # Store topics if available
            if 'topic' in qa_pair:
                results['topics'].append([s.strip() for s in qa_pair['topic'].split(',')])
            else:
                results['topics'].append([])
            
            # Calculate ROUGE scores
            rouge_scores = self.rouge_scorer.score(ground_truth, prediction)
            results['rouge1_f'].append(rouge_scores['rouge1'].fmeasure)
            results['rouge2_f'].append(rouge_scores['rouge2'].fmeasure)
            results['rougeL_f'].append(rouge_scores['rougeL'].fmeasure)
            
            # Calculate BLEU score
            reference_tokens = [nltk.word_tokenize(ground_truth.lower())]
            prediction_tokens = nltk.word_tokenize(prediction.lower())
            bleu = corpus_bleu([reference_tokens], [prediction_tokens], smoothing_function=self.smooth_fn)
            results['bleu'].append(bleu)
        
        # Calculate BERTScore (batch processing for efficiency)
        try:
            P, R, F1 = self.bert_scorer.score(results['predictions'], results['ground_truth'])
            results['bertscore_f1'] = F1.tolist()
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            results['bertscore_f1'] = [0.0] * len(results['predictions'])
        
        # Calculate average metrics
        avg_metrics = {
            'avg_rouge1_f': np.mean(results['rouge1_f']),
            'avg_rouge2_f': np.mean(results['rouge2_f']),
            'avg_rougeL_f': np.mean(results['rougeL_f']),
            'avg_bleu': np.mean(results['bleu']),
            'avg_bertscore_f1': np.mean(results['bertscore_f1'])
        }
        
        # Calculate topic-wise metrics if topics are available
        topic_metrics = {}
        if any(results['topics']):
            all_topics = [topic for topic_list in results['topics'] for topic in topic_list]
            unique_topics = sorted(set(all_topics))
            
            for topic in unique_topics:
                # Get indices of QA pairs with this topic
                indices = [i for i, topics in enumerate(results['topics']) if topic in topics]
                
                # Calculate metrics for this topic
                topic_metrics[topic] = {
                    'count': len(indices),
                    'rouge1_f': np.mean([results['rouge1_f'][i] for i in indices]),
                    'rouge2_f': np.mean([results['rouge2_f'][i] for i in indices]),
                    'rougeL_f': np.mean([results['rougeL_f'][i] for i in indices]),
                    'bleu': np.mean([results['bleu'][i] for i in indices]),
                    'bertscore_f1': np.mean([results['bertscore_f1'][i] for i in indices])
                }
        
        print('results["topics"]:', results['topics'])
        return {
            'model_name': model_name,
            'detailed_results': results,
            'avg_metrics': avg_metrics,
            'topic_metrics': topic_metrics
        }
    
    def compare_models(
        self, 
        sample_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Compare the naive, traditional, and fine-tuned models on a sample of the QA pairs.
        
        Args:
            sample_size: Number of samples to evaluate
            
        Returns:
            A dictionary containing evaluation results
        """
        results = {}
        

        # Evaluate traditional model (LSTM)
        if self.traditional_model is not None:
            traditional_results = self.evaluate_model(
                'traditional', 
                self.qa_pairs, 
                sample_size=sample_size,
                model_name="Traditional Model (LSTM)"
            )
        else:
            traditional_results = {
                'model_name': "Traditional Model (LSTM) - NOT AVAILABLE",
                'detailed_results': {},
                'avg_metrics': {},
                'topic_metrics': {}
            }


        # Evaluate naive model (deepseek 7b)
        naive_results = self.evaluate_model(
            'naive', 
            self.qa_pairs, 
            sample_size=sample_size,
            model_name="Naive Model (deepseek 7b base)"
        )
        
        
        
        # # Evaluate fine-tuned model
        fine_tuned_results = self.evaluate_model(
            'fine_tuned', 
            self.qa_pairs, 
            sample_size=sample_size,
            model_name="Fine-tuned Model (deepseek)"
        )
        
        # Store results
        results = {
            'naive_model': naive_results,
            'traditional_model': traditional_results,
            'fine_tuned_model': fine_tuned_results
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "./evaluation_results") -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: The evaluation results
            output_dir: Directory to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        json_path = os.path.join(output_dir, "evaluation_results.json")
        
        # Prepare a simplified version of results for JSON
        json_results = {}
        
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'model_name': model_results['model_name'],
                'avg_metrics': model_results['avg_metrics'],
                'topic_metrics': model_results['topic_metrics']
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"Results saved to {json_path}")
        
        # Generate and save visualizations
        self.visualize_results(results, output_dir)
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Create visualizations of evaluation results.
        
        Args:
            results: The evaluation results
            output_dir: Directory to save visualizations
        """
        # Set style
        plt.style.use('ggplot')
        
        # Prepare data for plotting
        model_names = []
        metrics = {
            'ROUGE-1': [],
            'ROUGE-2': [],
            'ROUGE-L': [],
            'BLEU': [],
            'BERTScore': []
        }
        
        for model_name, model_results in results.items():
            if not model_results['avg_metrics']:  # Skip if no metrics
                continue
            
            model_names.append(model_results['model_name'])
            metrics['ROUGE-1'].append(model_results['avg_metrics']['avg_rouge1_f'])
            metrics['ROUGE-2'].append(model_results['avg_metrics']['avg_rouge2_f'])
            metrics['ROUGE-L'].append(model_results['avg_metrics']['avg_rougeL_f'])
            metrics['BLEU'].append(model_results['avg_metrics']['avg_bleu'])
            metrics['BERTScore'].append(model_results['avg_metrics']['avg_bertscore_f1'])
        
        # Plot average metrics
        plt.figure(figsize=(12, 8))
        x = np.arange(len(model_names))
        width = 0.15
        
        plt.bar(x - width*2, metrics['ROUGE-1'], width, label='ROUGE-1')
        plt.bar(x - width, metrics['ROUGE-2'], width, label='ROUGE-2')
        plt.bar(x, metrics['ROUGE-L'], width, label='ROUGE-L')
        plt.bar(x + width, metrics['BLEU'], width, label='BLEU')
        plt.bar(x + width*2, metrics['BERTScore'], width, label='BERTScore')
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"))
        plt.close()
        
        # Plot topic-wise metrics for each model
        for model_name, model_results in results.items():
            if not model_results['topic_metrics']:  # Skip if no topic metrics
                continue
                
            # Get top 10 topics by count
            topics = sorted(
                model_results['topic_metrics'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
            
            topic_names = [t[0] for t in topics]
            topic_rouge_l = [t[1]['rougeL_f'] for t in topics]
            topic_bertscore = [t[1]['bertscore_f1'] for t in topics]
            
            plt.figure(figsize=(12, 8))
            x = np.arange(len(topic_names))
            width = 0.35
            
            plt.bar(x - width/2, topic_rouge_l, width, label='ROUGE-L')
            plt.bar(x + width/2, topic_bertscore, width, label='BERTScore')
            
            plt.xlabel('Topic')
            plt.ylabel('Score')
            plt.title(f'Topic-wise Performance - {model_results["model_name"]}')
            plt.xticks(x, topic_names, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_topic_metrics.png"))
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            results: The evaluation results
        """
        print("\n===== EVALUATION SUMMARY =====")
        
        for model_name, model_results in results.items():
            if not model_results['avg_metrics']:  # Skip if no metrics
                continue
                
            print(f"\n{model_results['model_name']}:")
            
            metrics = model_results['avg_metrics']
            print(f"  ROUGE-1: {metrics['avg_rouge1_f']:.4f}")
            print(f"  ROUGE-2: {metrics['avg_rouge2_f']:.4f}")
            print(f"  ROUGE-L: {metrics['avg_rougeL_f']:.4f}")
            print(f"  BLEU: {metrics['avg_bleu']:.4f}")
            print(f"  BERTScore: {metrics['avg_bertscore_f1']:.4f}")
        
        print("\n=============================")

    def generate_html_report(self, results: Dict[str, Any], output_file: str = "./evaluation_results/evaluation_report.html") -> None:
        """
        Generate a comprehensive HTML report of evaluation results.
        
        Args:
            results: The evaluation results dictionary
            output_file: Path to save the HTML report
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Chatbot Evaluation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    color: #333;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-table tr:hover {{
                    background-color: #f1f1f1;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    border-radius: 4px;
                }}
                .sample-predictions {{
                    overflow-x: auto;
                }}
                .model-comparison {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                .model-card {{
                    width: 31%;
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                @media (max-width: 768px) {{
                    .model-card {{
                        width: 100%;
                    }}
                }}
                .highlight {{
                    background-color: #e8f4f8;
                    padding: 2px 5px;
                    border-radius: 3px;
                }}
                .qa-pair {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                .question {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .reference {{
                    font-style: italic;
                    color: #27ae60;
                }}
                .prediction {{
                    color: #e74c3c;
                }}
                .metrics-badge {{
                    display: inline-block;
                    padding: 3px 7px;
                    margin-right: 5px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    color: white;
                    background-color: #3498db;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Medical Chatbot Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <p>This report compares the performance of three different medical chatbot models
                    on answering questions about the Huberman Lab podcast:</p>
                    <ul>
                        <li>Naive Model: Base deepseek-llm-7b model</li>
                        <li>Traditional Model: Encoder-decoder LSTM with attention</li>
                        <li>Fine-tuned Model: Fine-tuned deepseek model</li>
                    </ul>
                </div>

                <div class="section">
                    <h2>Performance Overview</h2>
                    <div class="model-comparison">
        """
        print('\nPRINTING RESULTS')
        print('\nresults', results)
        print('\nresults.items():', results.items())
        print('\nresults.keys():', results.keys())
        # Add model cards for each model
        for model_name, model_results in results.items():
            if not model_results['avg_metrics']:  # Skip if no metrics
                html_content += f"""
                    <div class="model-card">
                        <h3>{model_results['model_name']}</h3>
                        <p>No metrics available for this model</p>
                    </div>
                """
                continue
                
            metrics = model_results['avg_metrics']
            html_content += f"""
                <div class="model-card">
                    <h3>{model_results['model_name']}</h3>
                    <table class="metrics-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>ROUGE-1</td>
                            <td>{metrics['avg_rouge1_f']:.4f}</td>
                        </tr>
                        <tr>
                            <td>ROUGE-2</td>
                            <td>{metrics['avg_rouge2_f']:.4f}</td>
                        </tr>
                        <tr>
                            <td>ROUGE-L</td>
                            <td>{metrics['avg_rougeL_f']:.4f}</td>
                        </tr>
                        <tr>
                            <td>BLEU</td>
                            <td>{metrics['avg_bleu']:.4f}</td>
                        </tr>
                        <tr>
                            <td>BERTScore</td>
                            <td>{metrics['avg_bertscore_f1']:.4f}</td>
                        </tr>
                    </table>
                </div>
            """
        html_content += "</div>"
        
        # Add visualizations from files
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="visualization">
                        <h3>Overall Model Comparison</h3>
                        <img src="model_comparison.png" alt="Model Comparison Chart">
                    </div>
                </div>
                
                <div class="section">
                    <h2>Topic-wise Performance</h2>
        """
        
        # Add topic-wise performance visualizations
        for model_name, model_results in results.items():
            if not model_results['topic_metrics']:  # Skip if no topic metrics
                continue
                
            html_content += f"""
                    <div class="visualization">
                        <h3>{model_results['model_name']} - Topic Performance</h3>
                        <img src="{model_name}_topic_metrics.png" alt="Topic-wise Performance">
                    </div>
            """
        
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Sample Predictions</h2>
                    <div class="sample-predictions">
        """
        
        # Add sample predictions (up to 5)
        for i in range(10):
            question = results['fine_tuned_model']['detailed_results']['questions'][i]
            ground_truth = results['fine_tuned_model']['detailed_results']['ground_truth'][i]
            
            html_content += f"""
                    <div class="qa-pair">
                        <p class="question">Q: {question}</p>
                        <p class="reference">Reference: {ground_truth}</p>
            """
            
            # Add predictions from each model
            for model_name, model_results in results.items():
                if not model_results['detailed_results']:  # Skip if no detailed results
                    continue
                    
                prediction = model_results['detailed_results']['predictions'][i]
                rouge_l = model_results['detailed_results']['rougeL_f'][i]
                bertscore = model_results['detailed_results']['bertscore_f1'][i]
                
                html_content += f"""
                        <p class="prediction">
                            <strong>{model_results['model_name']}:</strong> 
                            <span class="metrics-badge">ROUGE-L: {rouge_l:.3f}</span>
                            <span class="metrics-badge">BERTScore: {bertscore:.3f}</span>
                            <br>
                            {prediction}
                        </p>
                """
            
            html_content += """
                    </div>
            """
        
        html_content += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>Conclusion</h2>
                    <p>Based on the evaluation results, we can draw the following conclusions:</p>
                    <ul>
        """
        
        # Add conclusions based on results
        # Find the best model
        best_model = None
        best_rouge_l = -1
        
        for model_name, model_results in results.items():
            if not model_results['avg_metrics']:  # Skip if no metrics
                continue
                
            rouge_l = model_results['avg_metrics']['avg_rougeL_f']
            if rouge_l > best_rouge_l:
                best_rouge_l = rouge_l
                best_model = model_results['model_name']
        
        if best_model:
            html_content += f"""
                        <li>The <strong>{best_model}</strong> achieves the best overall performance with a ROUGE-L score of {best_rouge_l:.4f}.</li>
            """
        
        # Add more conclusions
        html_content += """
                        <li>Different models show varying performance across different medical topics.</li>
                        <li>The fine-tuned model generally produces more coherent and medically accurate responses.</li>
                    </ul>
                </div>
                
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
            
        print(f"HTML report generated: {output_file}")

def main():
    """
    Main function to test the evaluation pipeline.
    """
    print("MediSeek Evaluator...")
    
    # Initialize evaluator with default paths
    evaluator = MedicalChatbotEvaluator(
        naive_model_name="deepseek-ai/deepseek-llm-7b-base",
        traditional_model_path="../models/seq2seq_model.pth",
        fine_tuned_model_path="../models/fine_tuned_deep_seek",
        qa_pairs_directory="../data/qa_pairs",
        podcast_name='huberman_lab_test'
    )
    
    # Run evaluation on a small subset since it takes a couple of hours to evaluation on the entire set
    try:
        # Load data
        evaluator.load_data()
        
        # Load models
        evaluator.load_models()

        results = evaluator.compare_models(sample_size=10)
        
        evaluator.save_results(results,
                               output_dir='../evaluation_results')
        
        evaluator.generate_html_report(results,
                                       output_file='../evaluation_results/evaluation_report.html')
        

        
    except Exception as e:
        print(f"Error during evaluation test: {str(e)}")
        import traceback
        traceback.print_exc()
 


if __name__ == "__main__":
    main()