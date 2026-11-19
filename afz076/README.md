# Traffic Sign Recognizer with Weather Stress Test (SignGuard)

A traffic sign classifier that is **tested against bad weather**, **knows when it is not sure**, and
**shows what it looked at**. Mid-term project for Professional Training (YN3012170034).

> Student: Sajad Afzali | Repository folder: `afz076`

## What it does
- Recognises the 43 sign classes of the German Traffic Sign Recognition Benchmark (GTSRB)
- **Weather simulation:** fog, night, rain, blur, noise and glare at 5 severity levels
- **Robustness study:** three architectures, each trained on clean images and with weather augmentation
- **Calibration:** temperature scaling so that "90% confident" really means about 90% correct
- **Reject option:** below a confidence threshold the app answers "Not sure" instead of guessing
- **Explainability:** Grad-CAM heatmap of the image regions behind each decision
- Gradio app: image upload, weather controls, model switching, threshold slider

## Research questions
1. How much does accuracy drop under each weather condition and severity?
2. Does training with weather augmentation help, including on a condition it never saw (glare) and on
   harder severity (5)?
3. Are the confidences trustworthy, and does temperature scaling fix them?
4. Can a reject option trade coverage for accuracy?

## Project structure
```
app.py                            Gradio app
src/                              classes, config, data, corruptions, models, trainer, calibration, gradcam, predict
training/train.py                 train one model (clean or weather-augmented)
evaluation/evaluate.py            clean + weather evaluation, calibration, reject analysis
evaluation/make_plots.py          figures for the report
evaluation/make_table.py          Markdown results table from robustness.csv
evaluation/results/, figures/     result files and figures (small, committed)
notebooks/SignGuard_Colab.ipynb   step-by-step Colab runner (trains and evaluates everything)
tests/                            pytest suite (runs on CPU with synthetic data, no download needed)
```

## Installation
```bash
git clone https://github.com/xahiru/PT22026.git
cd PT22026/afz076
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # optional, a few minutes
```

## Dataset (not stored in this repo)
GTSRB from Kaggle: <https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign>
(Stallkamp et al., 2012): 39,209 training and 12,630 test images, 43 classes. Download and unzip it so
that the folder contains `Train/` and `Test/` (and `Train.csv`, `Test.csv`). Pass that folder with
`--data-root`. In Colab the notebook downloads it automatically with `kagglehub`.

**Validation split.** The training set contains about 30 consecutive frames of every physical sign. We
split **by physical sign (track)**, not by image, so near-identical frames never appear in both training
and validation: 32,610 training and 6,599 validation images (220 distinct validation signs). The official
test set is used only for the final evaluation.

## Trained models (not stored in this repo)
Weights are too large for GitHub. Download the six trained models here and unzip into `outputs/`:

**https://drive.google.com/drive/folders/1nDSFJf88UsuScINdUgdvPL26UKFjeLUU?usp=sharing**

Or train them yourself (below). A free Colab T4 GPU needs about 10 minutes for ResNet-18, and under an
hour for all six models. The app only needs `outputs/<run>/model.pt` for the runs you want to try
(for example `resnet18_clean` and `resnet18_weather`).

## Run
```bash
# train (clean baseline and weather-augmented variant)
python training/train.py --data-root /path/to/gtsrb --model resnet18 --weather-p 0   --name resnet18_clean
python training/train.py --data-root /path/to/gtsrb --model resnet18 --weather-p 0.5 --name resnet18_weather
# other models: --model smallcnn --no-pretrained --epochs 15   |   --model convnext_tiny --epochs 6

# evaluate (also fits the temperature and stores it in the checkpoint), plot, print the table
python evaluation/evaluate.py --data-root /path/to/gtsrb --ckpt outputs/resnet18_clean/model.pt outputs/resnet18_weather/model.pt
python evaluation/make_plots.py
python evaluation/make_table.py

# app (set SIGNGUARD_CKPTS="Label=path,Label2=path2" to choose the models; default: every outputs/*/model.pt)
python app.py
```
`notebooks/SignGuard_Colab.ipynb` runs every step above on Colab.

## Models
| Name | Description | Pretrained |
|---|---|---|
| `smallcnn` | Our own CNN: 4 blocks of (conv-BN-ReLU x2, max-pool), global average pooling, dropout, linear head | no |
| `resnet18` | Residual network (He et al., 2016), new 43-way head | ImageNet (torchvision) |
| `convnext_tiny` | Modern ConvNet (Liu et al., 2022), new 43-way head | ImageNet (torchvision) |

Inputs are resized to 64x64. Training: AdamW, one-cycle learning-rate schedule, cross-entropy
(no label smoothing, so calibration can be studied separately), small rotation/shift jitter and
**no horizontal flip** because "turn left" and "turn right" would swap meaning.

## Methodology notes
- **Weather augmentation** uses fog, night, rain, blur and noise at severities 1-4. **Glare** and
  **severity 5** are never seen in training, so they measure generalisation.
- **Temperature scaling** (Guo et al., 2017) divides the logits by one learned number T, fitted on the
  clean validation set only. It changes confidence, not the predicted class.
- **Reject option:** if the top probability is below a threshold, the answer is "Not sure". We report
  coverage (share answered) against accuracy on the answered images.
- **Grad-CAM** (Selvaraju et al., 2017) weights the last convolutional feature maps by the gradient of the
  predicted class and shows the positive evidence as a heatmap.

## Results
Accuracy (%) on the official GTSRB test set, one training run per model. Weather columns are severity 5;
glare was never seen in training. Full numbers (severities 1, 3, 5, calibration, reject analysis) are in
`evaluation/results/`, figures in `evaluation/figures/`.

| Run | clean | fog | night | rain | blur | noise | glare |
|---|---|---|---|---|---|---|---|
| resnet18_clean | 98.5 | 59.1 | 25.6 | 40.9 | 92.2 | 41.0 | 15.8 |
| resnet18_weather | 98.8 | 89.6 | 64.1 | 90.0 | 96.1 | 86.9 | 23.0 |
| smallcnn_clean | 99.2 | 65.8 | 14.8 | 55.9 | 95.8 | 20.2 | 15.0 |
| smallcnn_weather | 98.9 | 92.9 | 67.1 | 96.5 | 97.2 | 89.7 | 24.5 |
| convnext_tiny_clean | 99.3 | 82.8 | 33.8 | 67.6 | 95.5 | 63.4 | 20.4 |
| convnext_tiny_weather | 99.4 | 94.1 | 66.6 | 94.8 | 97.2 | 88.9 | 27.0 |

Main observations:
- On clean images all six models reach 98.5-99.4%, so clean accuracy alone does not separate them.
- Models trained only on clean images lose a lot under strong weather (severity 5): fog 59-83%, night 15-34%,
  rain 41-68%, noise 20-63%. Weather-augmented training recovers most of it: fog 90-94%, night 64-67%,
  rain 90-97%, noise 87-90%. Blur hardly matters (92-97%).
- Glare, never seen in training, stays very hard for every model (15-27% at severity 5); weather training
  adds only about 7-10 points there.
- Our own small CNN, trained with weather augmentation, is as robust as the pretrained ResNet-18 and
  ConvNeXt-tiny, so the training data mattered more than the architecture here.
- Calibration: on clean test images all models are already well calibrated (ECE at most 0.006), so temperature
  scaling gives no consistent gain there (fitted T between 0.75 and 1.35). Under strong shift the confidence
  becomes unreliable (for example raw ECE 0.40 for convnext_tiny_clean under glare, severity 5).
- Each number comes from a single training run, so differences of 1-2 points should not be over-read.

## Troubleshooting
- If `python app.py` stops with a 502 error about localhost, a proxy or VPN is catching local traffic.
  Run `export NO_PROXY=127.0.0.1,localhost` first (Windows Command Prompt: `set NO_PROXY=127.0.0.1,localhost`).
- `pytest -q` runs about 60 tests on synthetic data; it can take several minutes on a slow CPU.

## Limitations
- Trained and tested on German signs only; signs in other countries differ
- Weather effects are simulated, not photographed; **real-world performance was not tested**
- 64x64 inputs lose detail; the system classifies a cropped sign and does not find signs in a street scene
- Calibration fitted on clean data may not stay accurate under heavy weather
- One training run per model (no confidence intervals)
- Educational project: not suitable for real driving decisions

## Acknowledgements and references
Pretrained ImageNet weights come from torchvision.


- Stallkamp et al., "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition", Neural Networks, 2012 (GTSRB)
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
- Liu et al., "A ConvNet for the 2020s", CVPR 2022
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
- Hendrycks and Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", ICLR 2019
