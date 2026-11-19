# Food-101 Hybrid Image Classifier

**Student Name:** Krishna Magar  
**Student ID:** 20243120001  
**Email:** krishnamagar0216@gmail.com  
**Course:** Professional Training 2 (Mid-term Assignment)  
**Instructor:** Ahamed Zahir 

---

This repo contains an interactive Gradio app and an evaluation pipeline for food image classification. The app classifies an uploaded image or a webcam capture into one of the 101 Food-101 categories using the pre-trained [`codealchemist01/food-image-classifier-hybrid`](https://huggingface.co/codealchemist01/food-image-classifier-hybrid) model. It then shows the top three predictions along with local nutrition information.

## Features

The app accepts either a file upload or a webcam capture. Predictions come from a hybrid ResNet50 + DeiT-Base model with an adaptive cross-attention fusion module, trained on the 101 Food-101 classes. For each image the interface shows the top three categories with confidence scores.

A local knowledge base in `food_info.py` provides calories per 100 g, common ingredients, and a short description for all 101 categories. If a predicted category is missing from that knowledge base, the app shows “Nutrition information not available” rather than failing. In practice, that fallback keeps the UI usable when the model predicts something outside the local data. Model download or load errors, invalid images, and unexpected model output shapes are also reported in the UI. The evaluation script produces a confusion matrix, a classification report, and JSON metrics including top-1, top-5, and per-class precision and recall.

## Project Structure

```text
food_classifier/
├── app.py                  # Gradio interface and main entry point
├── model_loader.py         # Model architecture, checkpoint loading, preprocessing
├── food_info.py            # Nutrition knowledge base (FOOD_INFO dictionary)
├── evaluate.py             # Food-101 evaluation and report generation
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes caches, weights, and eval_outputs/
├── README.md
└── eval_outputs/           # Created at runtime by evaluate.py
```

## Model Details

The model repository is [`codealchemist01/food-image-classifier-hybrid`](https://huggingface.co/codealchemist01/food-image-classifier-hybrid), and the model card lists it under Apache-2.0. The checkpoint is `best_model.pth`, downloaded automatically with `huggingface_hub`, and `real_class_mapping.json` maps output indices to Food-101 class names.

The architecture has two branches. ResNet50 produces 7×7 spatial features and a global vector, while DeiT-Base (distilled, patch size 16, 224×224 input) handles the transformer side. Fusion uses cross-attention in both directions, self-attention, and adaptive weighting of the global features. The classifier is a two-layer MLP with LayerNorm, GELU, and dropout. Inputs are 224×224 with ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). The model card reports about 82.5% validation top-1 accuracy and more than 95% top-5 accuracy. Your mileage may vary on a different validation run.

The exact architecture lives in `model_loader.py`. The loader also translates DeiT state-dict key names between `transformers` 4.x (`encoder.layer`, `attention.attention.query`, ...) and 5.x (`layers`, `attention.q_proj`, ...), which is what lets the checkpoint load across versions.

## Requirements

You need Python 3.10 or newer and the packages listed in `requirements.txt`.

## Installation

```bash
cd food_classifier
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the model checkpoint (about 1.2 GB) from the Hugging Face Hub. Later runs use the local Hugging Face cache.

## Running the App

```bash
python app.py
```

Open the local Gradio URL printed in the terminal, usually `http://127.0.0.1:7860`. Upload an image or use the webcam panel, then press **Analyze Food**. The top-3 predictions and nutrition panel appear alongside the image.
If CUDA is unavailable, or you want to force CPU inference, set `FOOD_CLASSIFIER_DEVICE=cpu` before launching:

```bash
# Linux / macOS
FOOD_CLASSIFIER_DEVICE=cpu python app.py

# Windows PowerShell
$env:FOOD_CLASSIFIER_DEVICE="cpu"; python app.py
```

Webcam access works on `localhost`. If you expose the app through a remote server, browsers generally require HTTPS for camera access.

## Running the Evaluation

```bash
python evaluate.py
```

By default the script evaluates a class-balanced sample of about 1000 images, roughly 9–10 per class. A few useful options:

```bash
python evaluate.py --max-images 500 --batch-size 8
python evaluate.py --max-images 500 --device cpu
python evaluate.py --max-images 500 --sample-mode head   # first 500 rows
python evaluate.py --full          # entire test split (slow)
```

Outputs go to `eval_outputs/`:

| File | Contents |
| `confusion_matrix.png` | 101×101 confusion matrix (seaborn heatmap) |
| `classification_report.txt` | Per-class precision, recall, F1, support |
| `eval_results.json` | Top-1/top-5 accuracy, per-class metrics, metadata |

Progress is printed to the console, for example `Processed 100/1000 images`.

The first evaluation run also downloads the Food-101 `validation` split (about 1.3 GB) from the Hugging Face Hub. Later runs use the local cache.

### Dataset and label alignment

The evaluation uses `load_dataset("ethz/food101", ...)` from the Hugging Face `datasets` library. That dataset exposes `train` and `validation` splits; the `validation` split is the official 250-images-per-class Food-101 test set. `evaluate.py` first tries `test` and falls back to `validation` with a warning.

The dataset’s integer label order is not exactly the model’s class order. For example, `cheesecake` and `cheese_plate` are swapped. The script therefore aligns labels by class name rather than raw index and logs a warning when the orders differ. If a dataset class is missing from the model mapping, those samples are skipped and counted in `num_skipped`.

By default the evaluation sample is stratified across classes instead of taking the first N rows. In practice, this tends to produce a more representative confusion matrix, since the dataset is grouped by class and the first 1000 rows would cover only a few classes and result in a misleading confusion matrix.

## Nutrition Knowledge Base

`food_info.py` defines `FOOD_INFO`, a dictionary keyed by Food-101 class name:

```python
"pizza": {
    "calories_per_100g": 266,
    "ingredients": ["pizza dough", "tomato sauce", "mozzarella", ...],
    "description": "Pizza is a baked flatbread ...",
}
```

Calories and ingredient lists are approximate and intended for educational and demo purposes. They are not medical or dietary advice.

## Limitations

The model only recognises the 101 Food-101 categories. Images outside that set will still receive a Food-101 label, sometimes with low confidence. The reported accuracy comes from the model card and may vary on the Hugging Face `validation` split. Nutrition values are generic per-100 g estimates and do not account for portion size, recipe, or preparation method. CPU inference is slow for this hybrid architecture; in my experience, a CUDA GPU is worth using for the full evaluation. Webcam capture requires browser camera permission and, for non-localhost deployments, HTTPS.

## Acknowledgements

The model is [`codealchemist01/food-image-classifier-hybrid`](https://huggingface.co/codealchemist01/food-image-classifier-hybrid) and the dataset is [`ethz/food101`](https://huggingface.co/datasets/ethz/food101). The Food-101 dataset comes from Bossard, Guillaumin, and Van Gool’s ECCV 2014 paper, *Food-101 – Mining Discriminative Components with Random Forests*. DeiT is from Touvron et al., *Training data-efficient image transformers & distillation through attention* (ICML 2021), and ResNet from He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016). The app is built with [Gradio](https://www.gradio.app/), [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), and [scikit-learn](https://scikit-learn.org/).
