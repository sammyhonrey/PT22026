# Text Summarization App

An interactive Python application that generates concise summaries from long English text. The app uses Hugging Face Transformer models with a Gradio interface and includes a script for evaluating summary quality with ROUGE metrics.

## Overview

Users paste article or document text into the web interface, select a summarization model and choose the desired summary length range. The application:

- Validates that enough text is provided before summarization.
- Adds the required task prefix when using T5.
- Truncates input to 1024 tokens.
- Generates summaries with beam search.
- Caches loaded models in memory so each model is loaded only once per session.
- Reports original word count, summary word count and approximate compression.

## Features

- Simple browser-based interface built with Gradio.
- Three configurable Hugging Face models:
  - DistilBART: a fast, balanced default option.
  - T5-small: the smallest and generally fastest option.
  - BART-large-CNN: a larger model that may produce more detailed summaries but uses more memory and time.
- Adjustable minimum and maximum summary length.
- Automatic input validation and user-readable errors.
- Model caching within the running process.
- ROUGE evaluation on a sample of the CNN/DailyMail test set.

## Model Comparison

| Model | Hugging Face identifier | Intended use |
|---|---|---|
| DistilBART | `sshleifer/distilbart-cnn-12-6` | General-purpose speed and quality balance |
| T5-small | `t5-small` | Limited-memory or faster experimentation |
| BART-large-CNN | `facebook/bart-large-cnn` | Higher-quality news-style summaries; slower and more memory-intensive |

Model files are downloaded from the Hugging Face Hub on first use and are not stored in this repository. The app currently configures Hugging Face downloads through `https://hf-mirror.com`.

## Project Structure

```text
dri064/
├── app.py               # Gradio interface and summarization pipeline
├── evaluate_model.py    # ROUGE evaluation on CNN/DailyMail samples
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .gitignore
└── .gitkeep
```

## Requirements

- Python 3.9 or newer.
- Internet access for downloading model weights and the evaluation dataset.
- Several hundred megabytes of free disk space for the selected model, depending on model size and cache state.
- PyTorch supported by the Python environment.
- A CPU works without additional setup, although GPU acceleration is significantly faster if PyTorch detects a compatible device.

## Installation

From the project directory:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Running the Application

Activate the virtual environment, then run:

```bash
python app.py
```

Gradio prints a local URL in the terminal. Open that URL in a browser. The application is also configured to create a temporary public `share=True` URL when Gradio and network access permit it.

### Using the Interface

1. Paste at least 30 words of English text.
2. Select a model.
3. Choose minimum and maximum summary lengths in words.
4. Select **Summarize**.
5. Review the summary and statistics below the result panel.

Minimum length must be smaller than maximum length. Summary lengths are converted to approximate token limits in the model generation code.

## Evaluation

The evaluation script assesses DistilBART against 20 examples from the CNN/DailyMail 3.0.0 test split:

```bash
python evaluate_model.py
```

It performs the following steps:

- Downloads the CNN/DailyMail dataset from Hugging Face.
- Loads `sshleifer/distilbart-cnn-12-6` with a Transformers summarization pipeline.
- Generates summaries with a maximum length of 100 tokens and a minimum length of 30 tokens.
- Compares predictions with the dataset's human-written highlights using ROUGE.
- Prints ROUGE results to the terminal.
- Writes the model name, sample count and scores to `results.txt`.

The sample size is intentionally limited to 20 to keep evaluation manageable on a laptop. For a stronger report, increase `N_SAMPLES` in `evaluate_model.py` and rerun the experiment with a fixed dataset split and generation settings.

The resulting `results.txt` file is an output artifact and may be regenerated at any time.

## Dataset

The evaluation script uses the [CNN/DailyMail 3.0.0 dataset](https://huggingface.co/datasets/cnn_dailymail), a collection of English news articles and human-written highlights commonly used for abstractive summarization research.

## How It Works

```text
User text
   ↓
Input validation
   ↓
Tokenizer (truncates to 1024 tokens)
   ↓
Hugging Face seq2seq Transformer
   ↓
Beam-search summary generation
   ↓
Decoded text + word statistics
```

T5 needs the `summarize:` instruction prefix, while the BART and DistilBART models do not. The app adds this prefix automatically based on the selected model identifier.

## Troubleshooting

**The model download fails or stalls**  
Confirm that internet access is available and allow more time for the first run. Hugging Face model files are downloaded only once and then reused from the local cache.

**A model reports being out of memory**  
Use T5-small or DistilBART, close other memory-intensive applications, or run in an environment with more RAM.

**Gradio shows an unexpected public URL error**  
The local interface can still be used through the URL printed in the terminal. If no public sharing service is needed, set `share=False` in `demo.launch(...)` inside `app.py`.

**The interface rejects short text**  
The app requires at least 30 words because very short passages do not provide enough information for meaningful summarization.

**Evaluation is slow**  
Model loading, CPU inference and dataset downloads can take time on a laptop. Reduce `N_SAMPLES` if faster exploratory runs are needed.

## Limitations

- Input is truncated when it exceeds the model's 1024-token processing limit.
- Summaries may contain hallucinated details that are not present in the source text.
- The models are mainly trained for English news and may perform poorly on poetry, code, technical documents or highly informal text.
- Word length is approximate because generation limits are converted from words to tokens.
- The app does not preserve document formatting or citation structure.
- CPU inference can be slow, especially for BART-large-CNN.
- Generated summaries should be reviewed before use in important or factual contexts.

## Future Improvements

- Add optional GPU device selection and progress indicators.
- Support PDF, TXT and DOCX file uploads.
- Allow custom generation settings such as temperature and beam count.
- Cache summaries and display model-loading status.
- Evaluate additional domain-specific datasets and models.
- Add automated tests for input validation and summary generation.

## Author

Student name: `<your name>`  
Student ID: `<your ID>`
