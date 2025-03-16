import os
import json
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekQAFinetuner:
    def __init__(
        self,
        model_name="deepseek-ai/deepseek-llm-1.3b-base",
        data_dir="./data/qa_pairs/huberman_lab",
        output_dir="./fine_tuned_models",
        device=None,
        load_in_8bit=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        max_seq_length=1024,
        batch_size=4,
        gradient_accumulation_steps=8,
        epochs=3,
        learning_rate=2e-5,
        save_steps=100,
        eval_steps=100,
        warmup_ratio=0.03,
    ):
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model configuration
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.load_in_8bit = load_in_8bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.warmup_ratio = warmup_ratio
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
    def load_model_and_tokenizer(self):
        """
        Load DeepSeek model and tokenizer
        """
        logger.info(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Add special tokens for instruction tuning if not present
        special_tokens = {"pad_token": "<pad>", "eos_token": "</s>", "bos_token": "<s>"}
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Load model with quantization if enabled
        if self.load_in_8bit:
            logger.info("Loading model in 8-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
        
        # Configure LoRA
        logger.info("Applying LoRA adapters")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def load_and_process_data(self, val_split=0.05):
        """
        Load QA pairs from JSON files and format them for instruction tuning
        """
        logger.info(f"Loading data from: {self.data_dir}")
        all_data = []
        
        # Process all JSON files
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.data_dir, file_name)
                try:
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    
                    if not data['qa_pairs']:
                        continue
                        
                    qa_pairs = data['qa_pairs']
                    df = pd.DataFrame(qa_pairs)
                    all_data.append(df)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
        
        # Combine all data
        if not all_data:
            raise ValueError("No valid data found in the specified directory")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} question-answer pairs")
        
        # Format data for instruction tuning
        formatted_data = []
        for _, row in combined_df.iterrows():
            # Format as instruction: question, response: answer
            formatted_text = f"<s>Human: {row['question']}\n\nAssistant: {row['answer']}</s>"
            formatted_data.append(formatted_text)
        
        # Split into training and validation sets
        np.random.seed(42)
        indices = np.random.permutation(len(formatted_data))
        val_size = int(len(formatted_data) * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_data = [formatted_data[i] for i in train_indices]
        val_data = [formatted_data[i] for i in val_indices]
        
        logger.info(f"Split data into {len(train_data)} training and {len(val_data)} validation examples")
        
        # Tokenize data
        logger.info("Tokenizing data")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
        
        # Create tokenized datasets
        train_encodings = tokenize_function(train_data)
        val_encodings = tokenize_function(val_data)
        
        class TextDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
                
            def __len__(self):
                return len(self.encodings.input_ids)
                
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = item["input_ids"].clone()
                return item
        
        train_dataset = TextDataset(train_encodings)
        val_dataset = TextDataset(val_encodings)
        
        return train_dataset, val_dataset
    
    def train(self):
        """
        Fine-tune the model on the QA dataset
        """
        # Load model and tokenizer if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model_and_tokenizer()
        
        # Load and process data
        train_dataset, val_dataset = self.load_and_process_data()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            load_best_model_at_end=True,
            report_to="tensorboard",
            fp16=self.device == "cuda",
            seed=42
        )
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        self.save_model()
        
        return trainer
    
    def save_model(self, model_path=None):
        """
        Save the fine-tuned model
        """
        if model_path is None:
            model_path = os.path.join(self.output_dir, "final_model")
        
        os.makedirs(model_path, exist_ok=True)
        logger.info(f"Saving model to {model_path}")
        
        # Save the model's PEFT adapters
        self.model.save_pretrained(model_path)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(model_path)
        
        # Save in PyTorch format for compatibility with the traditional model
        torch_path = os.path.join(model_path, "pytorch_model.pth")
        torch.save(self.model.state_dict(), torch_path)
        logger.info(f"Model state dict saved to {torch_path}")
        
        return model_path
    
    def generate_response(self, question, max_new_tokens=150, temperature=0.3):
        """
        Generate a response for a given question using the fine-tuned model
        """
        # Format the prompt
        prompt = f"<s>Human: {question}\n\nAssistant:"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate a response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        response = full_response.split("Assistant:")[1].strip()
        
        return response

def main():
    # Initialize and train DeepSeek model
    finetuner = DeepSeekQAFinetuner(
        model_name="deepseek-ai/deepseek-coder-1.3b-base",
        data_dir="../data/qa_pairs/huberman_lab",
        output_dir="./fine_tuned_models/deepseek",
        batch_size=4,  # Adjust based on your GPU memory
        epochs=3,
        learning_rate=2e-5,
        load_in_8bit=True  # Set to True if you have limited GPU memory
    )
    
    # Train the model
    trainer = finetuner.train()
    
    # Test the model with a sample question
    question = "How can omega-3 fatty acids benefit brain health?"
    response = finetuner.generate_response(question)
    print(f"Q: {question}")
    print(f"A: {response}")

if __name__ == "__main__":
    main()