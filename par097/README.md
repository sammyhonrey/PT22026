# Brain Tumor MRI Classification (par097)

Educational project for Professional Training (YN3012170034).
A web app that classifies brain MRI images into 4 classes (glioma, meningioma, no tumor, pituitary tumor)
and shows a Grad-CAM heatmap of where the model looked.

**This is not a medical diagnosis tool.**

## Model
- ResNet-18 pretrained on ImageNet, fine-tuned for 3 epochs (transfer learning).
- Test accuracy: 94.2% on 1,600 test images.
- Weights: https://huggingface.co/santaparvin/brain-mri-resnet18

## Dataset
Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle):
https://www.kaggle.com/datasets/masoudnickparvin/brain-tumor-mri-dataset
The dataset is not included in this repository.

## How to run
1. Install the libraries: `pip install -r requirements.txt`
2. Start the app: `python app.py` (the model weights download automatically from Hugging Face)
3. Open http://127.0.0.1:7860 and upload an MRI image.

## Train the model yourself
1. Download the dataset and unzip it so it contains the folders Training and Testing.
2. Run: `python train.py --data_dir "path/to/dataset" --epochs 3`

## Files
- `train.py` - fine-tunes ResNet-18 and prints the evaluation report
- `app.py` - Gradio web app with prediction and Grad-CAM
- `class_names.json` - class labels
- `requirements.txt` - libraries

## Limitations
The model confuses glioma and meningioma in some cases (glioma recall 0.835).
It was trained on one public dataset only and has no clinical validation.

## Acknowledgments
Dataset by Masoud Nickparvar (Kaggle); PyTorch, torchvision, Gradio and Hugging Face tools.
An AI assistant (Claude) was used for learning support and debugging.

## Sample images
`demo_images/` contains four example MRI images (one per class) taken from the
Brain Tumor MRI Dataset (Masoud Nickparvar, Kaggle) to try the app.