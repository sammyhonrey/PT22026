# Customer Review Classifier

Text classification project for Professional Training 2 (midterm).

## What it does
The app reads a customer review and says if it is POSITIVE or NEGATIVE.
It shows two models side by side so they can be compared:

1. **DistilBERT**, a ready-made model from Hugging Face
2. **TF-IDF + Logistic Regression**, my own model trained with scikit-learn

## Dataset
IMDB movie reviews (`stanfordnlp/imdb`): 10,000 reviews for training and
2,000 for testing. The dataset is downloaded automatically by `train.py`.

## Results
| Version | Settings | Accuracy |
|---|---|---|
| Model A | single words, stop words removed | 0.8645 |
| Model B | words and word pairs, stop words kept | 0.8765 |

Model B is slightly better. It also fixed the review "The product is not
good", which Model A got wrong because the word "not" was removed.

## Limitations
- Trained on movie reviews, not product reviews.
- Cannot detect sarcasm or mixed opinions.
- Only two classes: POSITIVE and NEGATIVE.

## How to run
1. Install Python 3.12.
2. Create and activate a virtual environment:
   ```
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```
3. Install the tools:
   ```
   pip install -r requirements.txt
   ```
4. Train my own model (creates `review_model.joblib`, which is not
   uploaded to GitHub because it is a large file):
   ```
   python train.py
   ```
5. Start the app:
   ```
   python app.py
   ```
6. Open http://127.0.0.1:7860 in your browser.

## Files
- `app.py`: the web app (Gradio)
- `train.py`: trains and tests my own model
- `requirements.txt`: the tools needed