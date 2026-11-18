# Pneumonia Detection from Chest X-Rays

## Overview
This project is an interactive machine learning application that classifies chest X-ray images as NORMAL or PNEUMONIA. It utilizes two fine-tuned deep learning models: ResNet-50 (a Convolutional Neural Network) and ViT-base (a Vision Transformer). 

To ensure clinical trust and address the black box problem, the application also integrates Grad-CAM (Gradient-weighted Class Activation Mapping) explainability. This provides a visual heatmap indicating exactly which parts of the X-ray the AI focused on to make its decision.

## Results
Both models were trained and evaluated using the validation set.

| Model | Best Validation Macro-F1 | Best Epoch |
|---|---|---|
| ResNet-50 | 0.984 | 3 |
| ViT-base | 0.975 | 3 |

## Tech Stack
- Language: Python
- Deep Learning: PyTorch, torchvision, Hugging Face Transformers
- Web Interface: Gradio
- Explainability: Grad-CAM

## Installation
1. Clone the repository and navigate to the project folder.
2. Install the required dependencies:
pip install -r requirements.txt

## Dataset
The dataset used for this project is the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
Download the dataset here: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract it so the folder structure matches the following:

data/chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
  test/
    NORMAL/
    PNEUMONIA/

## Model Weights
Due to GitHub file size limits, the trained model weights (.pt files) are excluded from this repository. 

To run the application, you must place the trained weights in the checkpoints/ directory:
- checkpoints/resnet50_best.pt
- checkpoints/vit_best.pt

## Usage
To launch the Gradio web application, run:
python app.py

Once the app starts, open http://127.0.0.1:7860 in your web browser. Upload a chest X-ray image, select a model, and click Analyze to see the prediction and heatmap.

## Training
To train the models from scratch, use the train.py script:
python train.py --model resnet50 --epochs 5
python train.py --model vit --epochs 5

## Project Structure

ull005/
  app.py
  gradcam.py
  model.py
  train.py
  plot_history.py
  utils.py
  requirements.txt
  README.md
  checkpoints/
  data/
  results/

## Limitations
- Educational Tool Only: This is a university project and not a certified medical device. It must not be used for real diagnosis or treatment.
- Dataset Bias: The model was trained on a single public dataset, meaning it may not generalize well to X-rays from different hospitals or imaging machines.
- Black Box: While Grad-CAM is used, we cannot 100% guarantee the model is looking at the exact clinical features a radiologist would look at.

## Author
Ullah Sami - 20243120005 