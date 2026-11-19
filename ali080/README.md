# Helmet Detection with YOLOv8

**Course:** Professional Training (PT22026) | **Student:** ALI REZAIE FATIMA | **Student ID:** 20243120080

An interactive web app that detects whether cyclists and riders in an image are wearing a helmet. It uses a **YOLOv8n** object detection model fine-tuned on a public helmet dataset, with a **Gradio** interface.

## Features
- Upload an image, paste one from the clipboard, or take a photo with the webcam (single-frame capture, not live video)
- Bounding boxes labelled **With Helmet** or **Without Helmet**
- Adjustable confidence threshold, with an option to show or hide confidence scores
- Class filter: show all detections, only With Helmet, or only Without Helmet
- Colour-coded safety banner (green: all riders wear helmets, red: warning, grey: nothing found or filter active)
- Results table with the class, confidence and box coordinates of each detection
- Example images that can be tried with one click
- Error handling for missing input and a missing model file

## Project structure
```
ali080/
├── app.py              # Gradio application
├── train.py            # Fine-tuning script (run on a GPU, e.g. Google Colab)
├── requirements.txt    # Python dependencies
├── models/             # Place best.pt here (not stored in Git)
├── examples/           # Sample test images
└── results/            # Training and evaluation charts
```

## Installation
Requires Python 3.9 or newer.

1. Clone the repository and open the project folder:
   ```bash
   git clone https://github.com/xahiru/PT22026.git
   cd PT22026/ali080
   ```
2. (Recommended) create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   pip install -r requirements.txt
   ```
3. Download the trained model and place it at `models/best.pt`:
   https://drive.google.com/file/d/1OKzWABCZ3r5EQg-5jPO32oMLYlRHvByb/view?usp=sharing

## Usage
```bash
python app.py
```
The app opens in your browser (usually at http://127.0.0.1:7860). Upload an image, take a webcam photo, or pick one of the examples, adjust the confidence threshold if you like, and click **Detect**.

## Model
- **Model:** YOLOv8n (Ultralytics), 3.1M parameters, 8.6 GFLOPs
- **Architecture:** a CSP-style backbone with C2f blocks extracts features, a PAN-FPN neck combines features at three scales, and an anchor-free decoupled head predicts class scores and box coordinates in separate branches.
- **Training loss:** box regression loss (CIoU), distribution focal loss (DFL) and classification loss.
- **Approach:** transfer learning. The COCO-pretrained `yolov8n.pt` weights were fine-tuned for 50 epochs (image size 640, batch 16) on a Google Colab T4 GPU (about 0.75 hours).

## Dataset
- **Bike Helmet Detection**, Roboflow Universe (workspace `bike-helmets`, version 2), **Public Domain** licence
- Classes: `With Helmet`, `Without Helmet`
- Provided with train / valid / test splits in YOLOv8 format
- The dataset is not included in this repository. `train.py` downloads it with the Roboflow API (set the `ROBOFLOW_API_KEY` environment variable).

## Results (test split, 63 images, 184 instances)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.856 | 0.832 | 0.884 | 0.423 |
| With Helmet | 0.910 | 0.911 | 0.950 | 0.490 |
| Without Helmet | 0.803 | 0.754 | 0.819 | 0.356 |

Charts (confusion matrix, PR curve, training curves) are in `results/`. They were produced on the validation split (126 images, 299 instances), while the table above uses the held-out test split.

## Limitations
- `Without Helmet` is detected less reliably than `With Helmet` (fewer training examples, harder to recognise).
- The dataset is small and shows cyclists, scooter riders and motorcyclists, mostly in daylight. Accuracy may drop with unusual angles, low light, heavy occlusion or very small riders.
- The webcam option analyses a single captured photo, not a live video stream.
- The small `n` model trades accuracy for speed.
- This is a coursework project and must not be used as a safety-critical system.

## Reproducing the training
Run `train.py` on a machine with an NVIDIA GPU (or Google Colab):
```bash
export ROBOFLOW_API_KEY="your_key"
python train.py --epochs 50 --imgsz 640 --batch 16
```

## Acknowledgements and references
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Gradio: https://www.gradio.app
- Dataset: Bike Helmet Detection by bike-helmets, Roboflow Universe (Public Domain)
- Roboflow: https://roboflow.com