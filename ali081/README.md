# ReviewLens — English Product Review Sentiment Analyzer

Author: Mohammad Alizada
Student ID: 20243120081
Course: Professional Training — YN3012170034
University: Yunnan University, School of Software
Assigned folder: ali081

## Overview

ReviewLens predicts positive or negative sentiment in English product
reviews using TF-IDF and Logistic Regression. A Gradio web interface
lets users analyze one review or a whole batch, see the probability
estimates, and see which words and phrases led to the prediction.

The app has three tabs:

1. **Single review** — verdict, probability gauge, highlighted words,
   and a chart of the strongest word contributions.
2. **Batch analysis** — many reviews at once, with charts, a results
   table, and a CSV download.
3. **About the model** — test accuracy, how the model works, and its
   limits.

## Installation

Run the following commands in a terminal inside the ali081 folder.
Use the same Python environment for installation, training, and the app.

Create an optional virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13.9, Gradio 6.17.3, scikit-learn 1.7.2, and
pandas 2.3.3. The saved model file depends on the scikit-learn version,
so use the versions in `requirements.txt`.

To run the notebook, Jupyter and matplotlib are also needed if they are
not already installed:

```bash
python -m pip install jupyter matplotlib
```

## Recreate the model

Datasets and trained models are excluded from Git.

On a fresh checkout, run:

```bash
python train_v2.py --output models/sentiment_pipeline_v2.joblib
```

The script downloads a sample of 20,000 review bodies from the official
Amazon Polarity training split, or reuses the local sample. It cleans
the data and trains the selected final model.

Internet access to Hugging Face is needed for the first download.
A GPU is not required.

The script refuses to overwrite an existing output model.
To reproduce separately when the application model already exists, run:

```bash
python train_v2.py
```

The default output is:

```text
models/sentiment_pipeline_v2_reproduced.joblib
```

A JSON file records package versions, training counts, and the raw
sample's SHA-256 hash.

If `huggingface.co` is slow or blocked on your network, set a mirror
before running the script. In PowerShell:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## Run the app

In the terminal, run:

```bash
python app.py
```

Open the local URL printed in the terminal (usually
`http://127.0.0.1:7860`).
Keep the terminal open while using the app. Press Ctrl+C to stop it.

The app loads `models/sentiment_pipeline_v2.joblib`. If the file is
missing, the app stops with a message that names it; create it with the
training command above.

## Using the app

**Single review**

1. Type or paste an English product review, or pick one of the examples.
2. Select **Analyze review**.
3. Read the verdict, the probability gauge, and the highlighted words.
   Hover over a highlighted word to see its score.
4. The **Recent analyses** box keeps the last six results.

**Batch analysis**

1. Paste one review per line, or upload a `.csv` or `.txt` file.
   For a `.csv`, use a column named `review`, `text`, `comment`, or
   `content` (case-insensitive); otherwise the first column is used. **Load sample reviews** fills in an example.
2. Select **Analyze batch**.
3. Read the summary charts and the results table, and download the
   full results as a CSV file.

## Features

- Positive and negative sentiment predictions with probability estimates.
- Verdict card, four metric boxes (positive %, negative %, signal
  strength, known words), and a probability gauge with an uncertain zone.
- Word highlighting: green words push toward positive, red words push
  toward negative. Faint contributors are left unmarked.
- Bar chart of the strongest single-word and two-word-phrase
  contributions.
- A warning when a review contains strong positive and strong negative
  wording (mixed signals).
- History of recent analyses.
- Batch analysis with a donut chart of the verdicts, a histogram of
  positive probability, the words that mattered most in the batch, a
  results table, and a CSV download (up to 500 reviews).
- Example reviews and Clear buttons.
- Helpful messages for empty, too long, or unrecognized input.
- A maximum input length of 5,000 characters, with a live counter.
- An uncertainty message when neither class reaches 60%.
- Light and dark mode, and a layout that adapts to narrow screens.

The 60% threshold is a display rule, not a calibrated confidence
threshold. The "signal strength" label (Weak, Moderate, Strong, Very
strong) is also only a display rule based on the top probability.

## How the model works

ReviewLens is a scikit-learn pipeline with two steps.

**1. TF-IDF features.** The review text is converted to lowercase and
split into single words and two-word phrases (`ngram_range=(1, 2)`).
Each term gets a weight that grows with how often it appears in the
review and shrinks with how many training reviews contain it:

```text
tf-idf(t, d) = tf(t, d) × idf(t)
idf(t)       = ln((1 + n) / (1 + df(t))) + 1
```

Here `n` is the number of training reviews and `df(t)` is the number of
reviews containing term `t`. With `sublinear_tf=True`, the term count
is replaced by `1 + ln(tf)`. Each review vector is then scaled to unit
length. Terms found in fewer than two reviews are ignored (`min_df=2`),
and at most 100,000 terms are kept.

**2. Logistic Regression.** The classifier learns one weight `w_t` for
every term and a bias `b`. For a review vector `x` it computes:

```text
P(positive | review) = 1 / (1 + exp(-(w · x + b)))
```

The probability of the negative class is `1 - P(positive)`. The
predicted class is the one with the larger probability.

**How the app explains a result.** Because the model is linear, the
contribution of each term to the score is its TF-IDF value multiplied
by its weight, `x_t × w_t`. Positive contributions raise the positive
probability and negative ones raise the negative probability. The app
highlights the strongest words and lists the strongest words and
phrases in the bar chart.

**Capabilities.** The model recognizes sentiment words and short
phrases such as "not good" or "the worst". It is fast, needs no GPU,
and its decisions can be explained word by word.

**Limits.** It treats a review as a bag of words and short phrases, so
it cannot follow longer-range meaning such as negation across several
words, sarcasm, or contrast between clauses. See the Limitations
section below.

## Version 1 experiment

The original experiment used the Amazon portion of UCI's Sentiment
Labelled Sentences dataset.

After normalized duplicate removal, 983 sentences remained:
688 training, 147 validation, and 148 test sentences.

Logistic Regression was selected using validation Macro F1 and retrained
on 835 training and validation sentences. Its validation accuracy was
82.99% and validation Macro F1 was 0.8298.

The saved baseline achieved 80.41% accuracy and 0.8036 Macro F1 on its
original 148-sentence test set. The bootstrap 95% confidence interval
for accuracy was 73.65% to 86.49%.

These results are recorded in `results/test_metrics.json` and
`results/test_classification_report.txt`. This small UCI test set is
separate from the shared Amazon test sample below.

## Version 2 experiment

A buffered-shuffle sample of 20,000 review bodies was obtained from
fancyzhx/amazon_polarity using seed 42 and buffer size 10,000.
This is not a uniform random sample of the complete dataset.

Seven duplicates were removed, leaving 19,993 reviews:
15,994 training and 3,999 validation reviews.

TF-IDF was fitted on training data only during model comparison.

| Model | Validation accuracy | Validation Macro F1 |
|---|---:|---:|
| Naive Bayes | 86.15% | 0.8613 |
| Logistic Regression | 88.00% | 0.8799 |

Logistic Regression was selected and retrained on all 19,993
development reviews. This validation comparison evaluates Naive Bayes
and Logistic Regression; the shared test comparison below evaluates
two saved Logistic Regression models trained with different data and settings.

Final settings:

- TF-IDF ngram_range=(1, 2), min_df=2.
- max_features=100000, sublinear_tf=True.
- No stop-word removal.
- Logistic Regression C=10, solver=liblinear.
- max_iter=2000, random_state=42.

## Shared test evaluation

Both versions were evaluated on the same 5,000-review sample from the
official Amazon Polarity test split, using shuffle seed 123 and
buffer size 10,000.

Duplicate and conflicting texts were checked. Exact normalized matches
with previously used reviews were excluded; none were found in this
sample. Near-duplicates were not exhaustively checked.

| Model | Test accuracy | Test Macro F1 |
|---|---:|---:|
| UCI baseline — Logistic Regression | 69.72% | 0.6936 |
| Version 2 — Logistic Regression | 87.56% | 0.8754 |

Version 2 correctly classified 4,378 of 5,000 reviews. A bootstrap 95%
confidence interval for its accuracy is 86.66% to 88.52%.

| Actual class | Predicted negative | Predicted positive |
|---|---:|---:|
| Negative | 2088 | 328 |
| Positive | 294 | 2290 |

| Version 2 class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 0.8766 | 0.8642 | 0.8704 |
| Positive | 0.8747 | 0.8862 | 0.8804 |

The gain was 17.84 percentage points on this shared sample.
Both saved classifiers are Logistic Regression. The training data,
feature settings, and classifier settings differ, so this comparison
does not isolate the effect of training data alone.

The training accuracy of Version 2 is much higher than its validation
accuracy, so some overfitting remains.

## Reproducibility

The standalone script reproduced all 5,000 saved model predictions in
the original environment using the existing local training sample.

This confirms prediction agreement on that sample, not byte-identical
files or identical behavior on every possible input.

Fresh downloads may differ if upstream data or sampling software changes.

`train_v2.py` recreates the selected final model. The notebook records
the exploratory comparisons and evaluation and can be run from top to
bottom; it needs internet access the first time to download the
datasets. Do not use the test sample to choose further model settings.

## Project files

| File or folder | Purpose |
|---|---|
| app.py | Gradio application (current version) |
| train_v2.py | Standalone model training |
| sentiment_analysis.ipynb | Development and evaluation |
| requirements.txt | Dependency versions |
| README.md | Project documentation |
| results/ | Evaluation tables and figures |
| screenshots/ | Application screenshots |
| batch_test.csv | Small batch input for functional testing |
| .gitignore | Excludes local datasets, models, and temporary files |
| data/ | Local datasets, excluded from Git |
| models/ | Local models, excluded from Git |

## Limitations

- Only positive and negative categories are supported.
- Neutral, mixed, sarcastic, and unrelated text may be misclassified.
- Negation can fail. In a diagnostic check, "This is not bad at all, I
  would buy it again." was classified as negative.
- Unrelated input is not automatically detected.
- Probability estimates are not calibrated guarantees.
- Confident predictions can still be wrong.
- Word highlights show which terms the model weighted, not what the
  review means.
- Amazon Polarity uses 1–2 stars as negative and 4–5 stars as positive.
  Three-star reviews are excluded.
- Review ratings may not represent every sentence's sentiment.
- Results describe the evaluated sample, not all product reviews.

## Sources

Datasets:

- UCI Sentiment Labelled Sentences:
  https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- Amazon Polarity dataset:
  https://huggingface.co/datasets/fancyzhx/amazon_polarity

References:

- Zhang, X., Zhao, J., and LeCun, Y. (2015). Character-level
  Convolutional Networks for Text Classification. Advances in Neural
  Information Processing Systems 28. https://arxiv.org/abs/1509.01626
- Kotzias, D., Denil, M., de Freitas, N., and Smyth, P. (2015). From
  Group to Individual Labels using Deep Features. Proceedings of the
  21st ACM SIGKDD Conference on Knowledge Discovery and Data Mining.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in
  Python. Journal of Machine Learning Research, 12, 2825–2830.
  https://arxiv.org/abs/1201.0490
- Abid, A., et al. (2019). Gradio: Hassle-Free Sharing and Testing of
  ML Models in the Wild. https://arxiv.org/abs/1906.02569

Software: scikit-learn, Gradio, pandas, NumPy, joblib, and the
Hugging Face `datasets` library.

## Submission

Keep project changes within ali081.
Do not commit downloaded datasets or trained model files.
Submit through the required pull-request workflow.
