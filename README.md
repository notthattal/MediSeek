# MediSeek: A Medical Chat Bot Trained on Huberman Lab Podcast Transcripts

---

## Project Overview

MediSeek is a deep learning-based chatbot designed to provide insightful answers to questions about health and wellness topics discussed in the Huberman Lab podcast. The chatbot supports three distinct models:

**Naive Model**: DeepSeek 7B, used without fine-tuning.

**Traditional Encoder-Decoder Model**: Attention-based LSTM trained on Huberman Lab podcast episodes.

**Fine-Tuned Model**: DeepSeek 7B fine-tuned on the Huberman Lab podcast episodes.

The goal is to evaluate the performance of these models and offer users the flexibility to choose the model that best suits their needs.

## Project Goals

- Develop a chatbot capable of answering health-related queries based on Huberman Lab podcast content.

- Provide users with multiple model options for varying levels of expertise and accuracy.

- Evaluate models using robust metrics to ensure reliability.

## Project Structure
```
MediSeek/
├── aws/
│ ├── setup-backend.sh
│ ├── test-integration.sh
│ └── upload-models.sh
├── data/
├── evaluation_results/
│ ├── .ipynb_checkpoints/
│ ├── evaluation_report.html
│ ├── evaluation_results.json
│ ├── fine_tuned_model_topic_metrics.png
│ ├── model_comparison.png
│ ├── naive_model_topic_metrics.png
│ └── traditional_model_topic_metrics.png
├── frontend/
├── models/
│ └── fine_tuned_deep_seek/
│ ├── adapter_config.json
│ ├── adapter_model.safetensors
│ ├── config.json
│ ├── fine_tuned_deep_seek.pth
│ ├── README.md
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ ├── tokenizer.json
│ └── seq2seq_model.pth
├── scripts/
│ ├── deep_learning_model/
│ │ ├── fine_tune_deep_seek.ipynb
│ │ └── fine_tune_deep_seek.py
│ └── traditional_model/
│ │ ├── __init__.py
│ │ └── traditional_model.py
│ ├── clean_podcasts.py
│ ├── evaluate_traditional.py
│ ├── evaluate.py
│ ├── podcast_qa_generator.py
│ └── podcast_scraper.py
├── server/
├── venv/
├── .gitignore
├── chatbot-config.txt
├── LICENSE
├── README.md
├── requirements.txt
└── setup-frontend.sh
```

## Data Collections
The dataset was created by scraping all available episodes of the Huberman Lab podcast using a custom script (podcast_scraper.py). Transcripts were extracted using Selenium-based scraping techniques from Podscribe's API. The raw transcripts were saved in `../data/raw_transcripts`

### Data Cleaning
Transcripts were cleaned using `clean_podcasts.py`:

- Removed metadata, timestamps, speaker numbers, and advertisements
- Standardized formatting to ensure clean textual data
- Saved cleaned transcripts in `../data/clean_transcripts`

### QA Pair Generation
Using GPT-4o (podcast_qa_generator.py), 30 question-answer pairs were generated per episode at varying levels of expertise:

- **General**: Questions for casual listeners

- **Specific**: Questions for health advocates

- **Technical**: Questions for medical professionals

Each QA pair includes metadata such as type and topic. The generated pairs are stored in JSON files in `../data/qa_pairs`

## Model Design

### Naive Model
The naive model uses DeepSeek 7B without any modifications or fine-tuning

### Traditional Encoder-Decoder Model 
Implemented as an attention-based LSTM (`traditional_model.py`):

- Encoder: Processes input queries into embeddings

- Decoder: Generates responses using attention mechanisms

- Trained on QA pairs from Huberman Lab podcasts

### Fine-Tuned DeepSeek Model
Fine-tuned DeepSeek 7B (`fine_tune_deep_seek.py`) using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation). Training involved:

- Instruction tuning with formatted QA pairs (e.g., "Human: [question]\n\nAssistant: [answer]").

- Optimization with gradient accumulation to handle large-scale data efficiently.

## Evaluation Strategy

### Test Set
The evaluation set consists of the latest five Huberman Lab podcast episodes not included in the training data

### Metrics
Models were evaluated using industry-standard metrics:

- **ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)**: Measures overlap between generated responses and references.

- **BLEU**: Evaluates precision of n-grams in generated responses

- **METEOR**: Considers semantic similarity between responses and references

- **BERTScore**: Uses contextual embeddings to assess response quality

### Visualization
Evaluation results are visualized as bar charts and distributions (`evaluate_traditional.py`), saved in `evaluation_results/`. The visualization are included in an HTML file found under the same directory

### Usage Intructions 

#### Setup

1. Clone this repository:
```bash
git clone <repo_url>
cd <repo_name>
```

2. Create a virtual environment & install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Windows use venv\Scripts\activate.bat
pip install -r requirements.txt
```

3. Download the model from the following link:
- [Download Models Folder](<https://drive.google.com/file/d/1h5c-C88fh7fx_R09EZ4ulD8NXYeohfVI/view>)
- Make sure the `.zip` is downloaded under `MediSeek/`
- Unzip the file using: 
**on mac OS**
```bash
   unzip models.zip -x "__MACOSX/*" "._*"
   rm models.zip
```

#### Data Preparation
Run the following scrips sequentially:
1. Scrape the podcast transcripts:
```bash
python scripts/podcast_scraper.py --transcript_dir ../data/raw_transcripts --output_dir ../data/raw_transcripts/huberman_lab
```

2. Clean transcripts:
```bash
python scripts/clean_podcasts.py
```

3. Generate QA pairs. Note that you need to create a .env file and include your openai key. This process takes 12 hours!
```bash
python scripts/podcast_qa_generator.py --transcript_dir ../data/clean_transcripts/huberman_lab --output_dir ../data/qa_pairs --podcast_name huberman_lab
```

#### Model Training
Train the traditional model:
```bash
python scripts/traditional_model/fine_tune_deep_seek.py
```

To Fine-Tune DeepSeek 7B, run the `scripts/deep_learning_model/fine_tune_deep_seek.ipynb` on Colab. It takes 6 hours on A100 GPU. Don't forget to upload the data found under `data/qa_pairs/huberman_lab`


#### Evaluation
To evaluate the traditional model on its own:
```bash
python scripts/evaluate_traditional.py
```

To evaluate all models side-by-side: 
```bash
python scripts/evaluate.py 
```
Note: This process takes about 1 hour and will need to be done on Colab with a GPU. 


## Results
Evaluation results are stored in ../evaluation_results as JSON files and HTML visualizations. Metrics include ROUGE, BLEU, METEOR, and BERTScore comparisons across all models.

## License
This project is licensed under MIT. 