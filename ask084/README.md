# Named Entity Recognition (NER) Studio

An interactive web application to detect and compare Named Entities in text using a BERT-based model and spaCy. This project was built as part of a Professional Training Mid-Term Project.

## 📖 Overview
This project provides a Gradio-based interface with two main features:
1. **Analyzer:** Highlight entities (PER, ORG, LOC, MISC) in text, view confidence scores, filter by type, and export results to CSV.
2. **Model Arena:** Run the same text through two different models (BERT and spaCy) and see where they agree and disagree, categorized by conflict type.

## ✨ Features
*   **BERT NER:** Uses the pre-trained `dslim/bert-base-NER` model for high-accuracy entity extraction.
*   **spaCy Comparison:** Maps `en_core_web_sm` OntoNotes labels onto the CoNLL scheme for direct comparison.
*   **Interactive Highlighting:** Color-coded entity highlighting in the browser.
*   **CSV Export:** Download your analysis results with one click.
*   **Robust Validation:** Friendly error messages for empty text or text exceeding 1,500 characters.

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/baqir-askary-se/PT22026.git
   cd PT22026/ask084

   ## Fine-tuned model

The project also includes a fine-tuned BERT model, trained on the CoNLL-2003 training split using Google Colab (T4 GPU). 
*   **Training script:** `notebooks/finetune_ner_colab.ipynb`
*   **How to use:** The app automatically detects the model if it is placed in `models/ner-finetuned/`.
*   **Results:** The fine-tuned model achieved an F1 of **0.901**, improving on the pre-trained baseline (0.888) and specifically improving PER detection (F1 0.951).