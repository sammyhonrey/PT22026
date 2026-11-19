# AgroGuard AI - Plant Disease Detector and Farming Assistant

Professional Training (YN3012170034) - Mid-term Project
Student: Hossain Md Pial | GitHub: hossainmdpial52 | Folder: `hos052`

AgroGuard AI is an interactive web application that detects plant leaf diseases from a photo and
gives simple farming advice. Its main idea is to make the model work on **real field photos**,
not only on clean laboratory photos.

## 1. Problem and idea

Most plant disease models are trained on the PlantVillage dataset, where every photo shows one leaf
on a plain background. Such models reach about 95-99% accuracy on lab photos, but they fail on photos
taken by farmers in the field (cluttered background, uneven light). A ready-made baseline model scored
**95.58%** on lab photos but only **31.78%** on field photos.

This project fine-tunes MobileNetV2 on lab photos plus real field photos with strong data augmentation,
compares three models, and builds an application around the best one.

## 2. Features

| Tab | What it does |
|---|---|
| **Diagnose** | Upload a leaf photo, choose a model, see the top-5 predictions, a disease report, care suggestions, a prevention checklist and a low-confidence warning |
| **Disease Library** | Browse the information page of all 38 classes (26 diseases + 12 healthy classes) |
| **Farming Chatbot** | Ask questions such as "How can I prevent Early Blight?". It remembers the last diagnosis for follow-up questions and never recommends specific pesticides or dosages |
| **Explain (Grad-CAM)** | Shows a heatmap of the image regions that influenced the model's decision (explainable AI) |

The Diagnose tab can switch between **AgroGuard Model 3 (ours)** and the **ready-made baseline**, so the
difference between lab-only and lab + field training can be shown live.

## 3. Results

Test accuracy (final epoch, no test-set selection):

| Model | Lab photos (PlantVillage test, 10,849 images) | Field photos (PlantDoc test, 236 images) |
|---|---|---|
| Ready-made baseline | 95.58% | 31.78% |
| Model 2 (ours, PlantVillage only) | 99.33% | 27.54% |
| **Model 3 (ours, PlantVillage + PlantDoc)** | 99.27% | **67.37%** |

Training more on lab photos (Model 2) does not help on field photos. Adding field photos (Model 3)
more than doubles the field accuracy with almost no loss on lab photos. More details are in
[`docs/results.md`](docs/results.md) and [`docs/project_log.md`](docs/project_log.md).

## 4. Model documentation

**Model:** MobileNetV2 (Sandler et al., 2018), ImageNet pre-trained (`google/mobilenet_v2_1.0_224`),
with a new 38-class output layer. About 2.3 million parameters (about 9 MB).

**How it works:**
- The image is resized to 256 pixels, center-cropped to 224 x 224 and normalized.
- MobileNetV2 is built from *depthwise separable convolutions* (a cheap filter per channel followed by a
  1 x 1 convolution that mixes channels) and *inverted residual blocks with linear bottlenecks*
  (a block expands the channels, filters them, then compresses them again, with a skip connection).
  This gives good accuracy with very few computations.
- The last layer outputs 38 scores. A softmax turns them into probabilities, and the top-5 are shown.

**Training (Google Colab, Tesla T4):** 12 epochs, batch size 64, AdamW (learning rate 5e-4, weight decay 1e-4),
cosine learning-rate schedule, cross-entropy with label smoothing 0.1, mixed precision (fp16).
Augmentation: random resized crop, horizontal and vertical flip, 30 degree rotation, color jitter.
Model 3 uses PlantVillage plus the PlantDoc field photos (repeated 6 times because that set is small).

**Explainability (Grad-CAM):** To show *why* the model decided, the app uses Grad-CAM (Selvaraju et al., 2017). For the predicted class, it takes the gradients of the class score with respect to the last convolutional feature maps (7 x 7 x 1280), averages them per channel to get an importance weight for each channel, adds up the weighted feature maps, keeps the positive part (ReLU), enlarges it to 224 x 224 and lays it over the photo. Red and yellow areas influenced the decision the most. In our tests the red area usually lies on the diseased part of the leaf, not on the background. The heatmap is coarse (7 x 7 grid), so it does not mark every spot exactly.

**Capabilities:** recognizes 38 classes (14 crops) from a single leaf photo, works much better on field photos
than a lab-only model, and runs on a normal laptop CPU.

**Limitations:**
- Field accuracy is 67.37%, much lower than lab accuracy. It must not replace an agricultural expert.
- The field test set has only 236 images, so small differences are not reliable.
- Weak classes on field photos: Corn Northern Leaf Blight vs Gray Leaf Spot, Potato Early vs Late Blight,
  healthy Tomato, Tomato Mosaic Virus, Tomato Bacterial Spot.
- Only the 38 classes of PlantVillage are supported; other crops and diseases are not recognized.
- It works on one leaf per photo and cannot find several diseases in one image.
- The knowledge base and the chatbot are rule-based, so they only answer questions about the supported diseases.
- Grad-CAM heatmaps have low resolution and show where the model looked, which is not a proof that the prediction is correct.

## 5. Project structure

```
hos052/
├── src/
│   ├── app.py              # Gradio interface (3 tabs)
│   ├── classifier.py       # model loading, preprocessing, prediction
│   ├── explain.py          # Grad-CAM heatmaps (explainable AI)
│   └── knowledge_base.py   # disease information, reports and chatbot logic
├── notebooks/
│   └── train_agroguard.ipynb   # training and evaluation notebook (Google Colab)
├── docs/
│   ├── results.md          # experiment results
│   └── project_log.md      # detailed project notes
├── models/                 # model weights (not stored in GitHub, see step 3 below)
├── data/                   # datasets (not stored in GitHub)
├── requirements.txt
└── .gitignore
```

## 6. Installation and usage

Requirements: Python 3.10 or newer (developed with 3.12) and an internet connection for the first run.

**Step 1: Get the code**
The project is on the branch `hos052-project` of my fork:
```
git clone -b hos052-project https://github.com/hossainmdpial52/PT22026.git
cd PT22026/hos052
```
After the Pull Request is merged, the same folder `hos052` is also in the main course repository
(https://github.com/xahiru/PT22026).

**Step 2: Install the libraries**
```
pip install -r requirements.txt
```

**Step 3: Download the trained model (Model 3)**
Model weights are not stored in GitHub (course rule), so download them from Google Drive:

https://drive.google.com/drive/folders/1UUDsUV2dfipey_0GkEDTA39S0c2IYRLv?usp=sharing

Download the folder (it comes as a zip file), extract it, and put the two files `config.json` and
`model.safetensors` **directly** inside the folder `models/model3`, so that you have:
```
hos052/models/model3/config.json
hos052/models/model3/model.safetensors
```
Without these files, the "ready-made baseline" model still works, but "AgroGuard Model 3" shows a clear error message.

**Step 4: Run the application**
```
python src/app.py
```
Open the address shown in the terminal (normally http://127.0.0.1:7860) in your browser.
The first run also downloads the small ready-made baseline model from the Hugging Face Hub (no login needed).

**Using the app:** open the **Diagnose** tab, upload a leaf photo, select a model and press **Diagnose**.
Use **Disease Library** to read about a disease, **Farming Chatbot** to ask questions, and **Explain (Grad-CAM)** to see where the model looked.

## 7. Datasets (not stored in GitHub)

| Dataset | Where to get it | Use |
|---|---|---|
| PlantVillage | Hugging Face: `BrandonFors/Plant-Diseases-PlantVillage-Dataset` (43,456 train / 10,849 test images, 38 classes) | Training and lab-photo testing |
| PlantDoc | https://github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0; 2,342 train / 236 test images, 27 classes) | Field photos for training and testing |

The notebook downloads both datasets automatically.

## 8. Reproducing the training

Open `notebooks/train_agroguard.ipynb` in Google Colab, select a T4 GPU (Runtime > Change runtime type),
and run the cells from top to bottom. The last cell trains Model 2 and Model 3 (several hours) and saves
checkpoints, plots and the final models to Google Drive. If Colab disconnects, run the last cell again; it resumes.

## 9. Acknowledgements and references

- Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018 (arXiv:1801.04381)
- Hughes and Salathe, "An open access repository of images on plant health to enable the development of mobile
  disease diagnostics" (PlantVillage), 2015 (arXiv:1511.08060)
- Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection", 2020 (arXiv:1911.10317)
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017 (arXiv:1610.02391)
- Gildenblat and contributors, "PyTorch library for CAM methods" (pytorch-grad-cam), https://github.com/jacobgil/pytorch-grad-cam
- Ready-made baseline model: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (Hugging Face)
- Base model: `google/mobilenet_v2_1.0_224` (Hugging Face)
- Libraries: PyTorch, torchvision, Hugging Face Transformers and Datasets, Gradio, scikit-learn, matplotlib

## 10. Disclaimer

AgroGuard AI is a support tool for learning and early guidance. Always consult a local agricultural
extension officer before making treatment decisions. The system does not suggest pesticides or dosages.
