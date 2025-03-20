import os
import re

def clean_transcript(text):
    """
    Comprehensive function to clean a podcast transcript by:
    1. Removing header metadata
    2. Removing ad sections
    3. Removing timestamps and speaker numbers
    
    Returns clean transcript content only.
    """
    # Step 1: Remove header metadata
    # Find the first timestamp to identify where the actual content begins
    match = re.search(r'\d+\s+\d{2}:\d{2}:\d{2}', text)
    if match:
        # Get index of the first timestamp
        first_timestamp_index = match.start()
        # Find the line start before this timestamp
        header_end = text.rfind('\n', 0, first_timestamp_index) + 1
        # Remove everything before that point
        text = text[header_end:]
    
    # Step 2: Remove ad sections
    # Pattern looks for: lines with "(Ad)" and all content until the next section header
    pattern = r'.*\(Ad\).*\n(.*\n)*?(?=\d+\s\s|\Z)'
    text = re.sub(pattern, '', text)
    
    # Step 3: Clean up formatting
    # Remove timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    
    # Remove speaker numbers (standalone digits at line start)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

    text = text.lower()
    

    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    
    return text.strip()

def process_file(input_file, output_file):
    """Process a transcript file and save the cleaned version."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cleaned = clean_transcript(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print(f"Cleaned transcript saved to {output_file}")

def main():
    podcast_name = "huberman_lab"
    input_folder = "../data/raw_transcripts"
    output_folder = "../data/clean_transcripts"
    output_folder = os.path.join(output_folder, podcast_name)

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            process_file(input_file, output_file)

# Example usage
if __name__ == "__main__":
    main()