# SMS Spam Detection

A machine learning project that reads an SMS message and decides if it is **spam** or **not spam (ham)**. It includes a training script that compares three models and a small web app built with Gradio.

**Author:** Salsabile
**Course:** Professionaleeee Training (intermediate level), midterm individual project

## Dataset

SMS Spam Collection (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

- 5,574 English SMS messages: 4,827 ham and 747 spam
- The dataset is **not** included in this repository.

Reference: Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering (DocEng '11).

## How it works

1. Read the messages and their labels.
2. Split the data: 80% to train, 20% to test.
3. Turn the text into numbers with TF-IDF.
4. Train three models: Naive Bayes, Logistic Regression, Linear SVM.
5. Compare them with accuracy, precision, recall, and F1-score.
6. Save the best model (by F1-score) to `model.joblib`.
7. The Gradio app loads the saved model and classifies new messages.

## Results (20% test set)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.9731 | 1.0000 | 0.7987 | 0.8881 |
| Logistic Regression | 0.9785 | 0.9032 | 0.9396 | 0.9211 |
| **Linear SVM** | **0.9839** | 0.9517 | 0.9262 | **0.9388** |

Best model: **Linear SVM**.

## Project files

| File | What it does |
|---|---|
| `train.py` | Trains and compares the models, saves the best one |
| `app.py` | Web app (Gradio) to test messages |
| `requirements.txt` | Python packages needed |
| `results.csv` | Table of results |
| `confusion_matrix.png` | Picture of the best model's mistakes |

## How to run (Windows, Python 3.12)

1. Clone the repository and go into the project folder:
   ```
   cd PT22026\sid066
   ```
2. Create and activate a virtual environment:
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Download the dataset from the link above, unzip it, and put the file `SMSSpamCollection` in a folder called `data`:
   ```
   data/SMSSpamCollection
   ```
5. Train the models:
   ```
   python train.py
   ```
6. Start the app:
   ```
   python app.py
   ```
   Then open http://127.0.0.1:7860 in your browser.

## Limitations

- The data is old and mostly British and Singaporean English. New spam styles may be missed.
- The model only works for English messages.
- The app shows a confidence score, not a true probability.
- Short or unusual messages can be classified wrongly.