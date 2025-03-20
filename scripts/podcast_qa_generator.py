'''
This script implements PodcastQAGenertor, which utilizes gpt4o to generate question-answer pairs that
will be fed into the models (both for traditional and the fine-tuned LLM). 
Took 12 hours to generate 30 QA pairs per podcast episode (total of 384 episodes)
'''
import os
import json
import time
from openai import OpenAI
from tqdm import tqdm
import argparse
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize
import dotenv 

dotenv.load_dotenv()
# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PodcastQAGenerator:
    def __init__(self, output_dir="qa_pairs", model="gpt-4o", max_tokens=8000):
        """
        Initialize the QA Generator
        
        Args:
            output_dir (str): Directory to save generated QA pairs
            model (str): OpenAI model to use
            max_tokens (int): Maximum tokens for response
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # Set up a .env and add api key to run
        self.output_dir = output_dir
        self.model = model
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model(model)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def count_tokens(self, text):
        """Count the number of tokens in a text"""
        return len(self.encoder.encode(text))
    
    def chunk_transcript(self, transcript, max_input_tokens=6000):
        """
        Split transcript into chunks that fit within token limits
        
        Args:
            transcript (str): Full podcast transcript
            max_input_tokens (int): Maximum tokens per chunk
            
        Returns:
            list: List of transcript chunks
        """
        # First try sentence-based chunking to preserve context
        sentences = sent_tokenize(transcript)
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If this sentence alone exceeds the limit, we need to split it
            if sentence_tokens > max_input_tokens:
                # Add the current chunk if it's not empty
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                
                # Split the long sentence into smaller parts
                words = sentence.split()
                temp_sentence = []
                temp_count = 0
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    if temp_count + word_tokens > max_input_tokens:
                        chunks.append(" ".join(temp_sentence))
                        temp_sentence = [word]
                        temp_count = word_tokens
                    else:
                        temp_sentence.append(word)
                        temp_count += word_tokens
                
                if temp_sentence:
                    current_chunk.append(" ".join(temp_sentence))
                    current_token_count = self.count_tokens(" ".join(current_chunk))
            
            # If adding this sentence would exceed the limit
            elif current_token_count + sentence_tokens > max_input_tokens:
                # Save the current chunk and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_tokens
            else:
                # Add the sentence to the current chunk
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def generate_qa_from_chunk(self, chunk, podcast_name, chunk_index, total_chunks):
        """
        Generate QA pairs from a single transcript chunk
        
        Args:
            chunk (str): Transcript chunk
            podcast_name (str): Name of the podcast
            chunk_index (int): Index of this chunk
            total_chunks (int): Total number of chunks
            
        Returns:
            list: Generated QA pairs for this chunk
        """
        # Adjust number of questions based on chunk size and position
        num_questions = max(5, min(10, 30 // total_chunks))
        
        system_prompt = f"""
        You are an expert in generating high-quality question-answer pairs from medical podcast transcripts.
        You're working with chunk {chunk_index+1} of {total_chunks} from the podcast "{podcast_name}".
        
        Your task is to create {num_questions} diverse question-answer pairs based on the transcript chunk provided.
        
        Create a mix of question types:
        1. General questions that the public might ask out of curiosity or for surface-level help that would guide them in the right direction to research health related subjects more
        2. More specific questions that show deeper understanding, probably ask by health advocates and people wanting to optimize their lifestyles by listening to health podcasts 
        3. Technical questions that medical professionals might ask
        
        Focus on the most important and unique information in this chunk.
        Make answers factual and based only on the transcript content.
        
        Format your response as a JSON array with the following structure:
        [
            {{
                "question": "Question text here?",
                "answer": "Answer text here.",
                "type": "general" // or "specific" or "technical",
                "topic": "1 to 3 topic names that could help identify the topic of the answer"
            }},
            ...
        ]
        """
        
        user_prompt = f"""
        TRANSCRIPT CHUNK:
        {chunk}
        
        Based on this transcript chunk, generate {num_questions} high-quality question-answer pairs as instructed.
        """
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3, # Lower temp for more accuracy
                max_tokens=self.max_tokens
            )
            
            # Extract and parse JSON response
            json_content = response.choices[0].message.content
            # Sometimes the model returns markdown-wrapped JSON
            json_content = json_content.replace("```json", "").replace("```", "").strip()
            qa_pairs = json.loads(json_content)
            
            return qa_pairs
            
        except Exception as e:
            print(f"Error processing chunk {chunk_index+1}/{total_chunks}: {str(e)}")
            time.sleep(5)  # Wait before retrying
            return []
    
    def generate_qa_pairs(self, transcript_path, podcast_name, podcast_date=None):
        """
        Generate QA pairs from a podcast transcript using GPT-4o
        
        Args:
            transcript_path (str): Path to the transcript file
            podcast_name (str, optional): Name of the podcast for better context
            
        Returns:
            dict: Generated QA pairs
        """
        # Extract podcast date since the files are named based on date
        if podcast_date is None:
            podcast_date = os.path.basename(transcript_path).split('.')[0]
            
        # Load transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Check token count
        total_tokens = self.count_tokens(transcript)
        print(f"Transcript has {total_tokens} tokens")
        
        # Split into chunks if needed
        if total_tokens > 6000:
            chunks = self.chunk_transcript(transcript)
            print(f"Split transcript into {len(chunks)} chunks")
            
            # Process each chunk and collect QA pairs
            all_qa_pairs = []
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)} ({self.count_tokens(chunk)} tokens)")
                chunk_qa_pairs = self.generate_qa_from_chunk(chunk, podcast_name, i, len(chunks))
                all_qa_pairs.extend(chunk_qa_pairs)
                # Avoid rate limits
                time.sleep(2)
        else:
            # Process the entire transcript at once
            all_qa_pairs = self.generate_qa_from_chunk(transcript, podcast_name, 0, 1)
        
        # Add metadata
        result = {
            "podcast_name": podcast_name,
            "transcript_path": transcript_path,
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qa_pairs": all_qa_pairs
        }
        
        # Save to file
        output_path = os.path.join(self.output_dir, f"{podcast_name}_{podcast_date}_qa_pairs.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
            
        print(f"Generated {len(all_qa_pairs)} QA pairs for {podcast_name}")
        return result
    
    def batch_generate(self, transcript_dir, podcast_name, file_extension='.txt'):
        """
        Generate QA pairs for all transcript files in a directory
        
        Args:
            transcript_dir (str): Directory containing transcript files
            podcast_name (str): Directory for the specific podcast name
            file_extension (str): File extension of transcript files
            
        Returns:
            list: Generated QA pairs for all transcripts
        """
        transcript_files = [f for f in os.listdir(transcript_dir) 
                          if f.endswith(file_extension)]
        
        results = []
        for file in tqdm(transcript_files, desc="Generating QA pairs"):
            podcast_date = os.path.basename(file).split('.')[0]
            transcript_path = os.path.join(transcript_dir, file)
            json_file_name = f"{podcast_name}_{podcast_date}_qa_pairs.json"
            output_path = os.path.join(self.output_dir, json_file_name)
            try:
                # should only run if the file has not been processed yet 
                if os.path.exists(output_path):
                    print(f'The transcript at {file} has already been processed into QA pairs')
                    print(f'Skipping....\n')
                    continue
                else:
                    result = self.generate_qa_pairs(transcript_path, podcast_name)
                if result:
                    results.append(result)
                # Sleep to avoid hitting API rate limits
                time.sleep(5)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                # Continue with next file instead of stopping
                continue
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate QA pairs from podcast transcripts using GPT-4o")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", default="qa_pairs", help="Directory to save generated QA pairs")
    parser.add_argument("--file_extension", default=".txt", help="File extension of transcript files")
    parser.add_argument("--podcast_name", required=True, help="File extension of transcript files")
    
    args = parser.parse_args()
    print('\nArguments received:')
    print(f'transcript_dir = {args.transcript_dir}')
    print(f'output_dir = {args.output_dir}')
    print(f'file_extension = {args.file_extension}')
    print(f'podcast_name = {args.podcast_name}')

    '''
    python podcast_qa_generator.py --transcript_dir '../data/clean_transcripts/huberman_lab' --output_dir '../data/qa_pairs' --podcast_name huberman_lab
    '''

    output_dir = os.path.join(args.output_dir, args.podcast_name)
    generator = PodcastQAGenerator(output_dir=output_dir)
    generator.batch_generate(args.transcript_dir, file_extension=args.file_extension, podcast_name=args.podcast_name)