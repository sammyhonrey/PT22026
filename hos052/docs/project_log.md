# AgroGuard AI - Project Log (single source of truth for the report)

This file records every important fact, decision and result of the project, so the
technical report can be written from it at any time without relying on chat history.

## 1. Basic information
- Project title: AgroGuard AI - Advanced Plant Disease Detection & Smart Farming Assistant
- Course: Professional Training (course code YN3012170034), Mid-term project, individual
- Student: Hossain Md Pial | GitHub: hossainmdpial52 | Repository folder: hos052
- Submission: Pull Request from the student's fork to https://github.com/xahiru/PT22026
  (TA instruction: fork the repo, work inside the hos052 folder of the fork, then open a PR)
- Required deliverables: source code via PR, PDF report (min. 10 pages), 3-5 minute live demo
- Marks: App 30, Code + GitHub 20, Report 30, Depth/Creativity 10, Demo 10

## 2. Problem statement (the main story of the report)
Plant disease models trained on lab images (PlantVillage: single leaf, plain background)
reach about 95-99% accuracy on lab images but fail on real field photos (cluttered
background, uneven light). A ready-made model scored 95.58% on lab images but only 31.78%
on field images. This project fine-tunes MobileNetV2 with field images and strong
augmentation, compares three models, and wraps the best one in a farming assistant app.

## 3. Datasets
| Dataset | Source | Size | Use |
|---|---|---|---|
| PlantVillage | Hugging Face: BrandonFors/Plant-Diseases-PlantVillage-Dataset | 43,456 train / 10,849 test, 38 classes | Main training data, lab images |
| PlantDoc | GitHub: pratikkayal/PlantDoc-Dataset (CC BY 4.0), paper: arXiv 1911.10317 | 2,342 train / 236 test, 27 classes (spider mites class with 2 images excluded) | Field images for training and for the field test set |
- All 27 PlantDoc classes were mapped to PlantVillage class names by keyword rules (27 of 27 matched).
- Observation: PlantVillage sample images show one leaf on a plain gray background.

## 4. Models
- Ready-made baseline: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (MobileNetV2, PlantVillage, 38 classes).
  Its class order was verified to be identical to the dataset class order (ids 0-37).
- Model 2 (ours): MobileNetV2, ImageNet pre-trained (google/mobilenet_v2_1.0_224), trained on PlantVillage only.
- Model 3 (ours, used in the app): same architecture, trained on PlantVillage + PlantDoc train (PlantDoc repeated 6 times).
- Training setup (identical for Model 2 and 3, so the comparison is fair):
  12 epochs, batch 64, AdamW (lr 5e-4, weight decay 1e-4), cosine LR schedule,
  cross-entropy with label smoothing 0.1, mixed precision fp16, Tesla T4 GPU on Google Colab.
- Augmentation: RandomResizedCrop(224, scale 0.4-1.0), horizontal and vertical flip,
  rotation 30 degrees, color jitter (brightness/contrast/saturation 0.4, hue 0.1).
- Evaluation preprocessing: resize 256, center crop 224, normalize mean 0.5 / std 0.5.
- Reported numbers come from the final (12th) epoch, not the best epoch (no test-set peeking).

## 5. Results (test accuracy)
| Model | Lab images (PlantVillage test, 10,849) | Field images (PlantDoc test, 236) |
|---|---|---|
| Ready-made baseline | 95.58% | 31.78% |
| Model 2 (PlantVillage only) | 99.33% | 27.54% |
| Model 3 (PlantVillage + PlantDoc) | 99.27% | 67.37% |

Observations for the Discussion section:
1. Model 3 more than doubled field accuracy (31.78% to 67.37%) with almost no loss on lab images.
2. Model 2 reached 99.33% on lab images but only 27.54% on field images: more lab data/training does not fix the lab-to-field domain gap.
3. Training curves: Model 3 starts at about 62% field accuracy at epoch 1 and stays between 60% and 69%;
   Model 2 fluctuates between 25% and 35% (best about 35% at epoch 2).
4. Lab accuracy of both models is near 99%, a ceiling effect, since PlantVillage images are very uniform.
5. PlantDoc test has only 236 images (one image is about 0.42%), so small differences are not reliable;
   per-class counts are 4-12 images, so per-class numbers are noisy.
6. Confusion matrix (Model 3, field): strong on Grape healthy, Tomato Septoria leaf spot, Apple Cedar rust,
   Strawberry healthy, Corn common rust, Tomato late blight. Weak on Corn Northern Leaf Blight vs Gray leaf spot,
   Potato early vs late blight, Tomato healthy, Tomato mosaic virus, Tomato bacterial spot (often confused with Septoria).
7. Limitation: PlantDoc has 27 classes only; some PlantVillage classes (e.g. orange, squash types) have no field images.
8. Limitation: field accuracy of 67% is still far from lab accuracy; it must not replace an agricultural expert.

Demo examples (same field photo of a tomato leaf with late blight):
- Ready-made baseline: Apple Scab 24%, Squash Powdery Mildew 22%, Tomato Late Blight only 15% (wrong).
- Model 3: Tomato - Late blight 96% (correct). Caution: this photo may be similar to the training data
  (PlantDoc is also web-scraped), so 96% is an example, not the true accuracy. The true unseen accuracy is 67.37%.
- Lab-style photo (single leaf, plain background): ready-made baseline predicted Tomato Late Blight 72% (correct).
Interpretation: the baseline works on lab-style photos and fails on field photos.

## 6. Application (Gradio, Python)
Tabs:
1. Diagnose: upload leaf image, choose a model (Model 3 or ready-made baseline), top-5 predictions,
   a report with disease information, plant-care suggestions, prevention checklist, safety note,
   and a warning when confidence is below 50%.
2. Disease Library: dropdown of all 38 classes with the same report.
4. Explain (Grad-CAM): upload a leaf image, choose a model, get a heatmap of the regions that influenced the top
   prediction (src/explain.py, library pytorch-grad-cam, target layer = last 1x1 convolution of MobileNetV2, 7x7 map).
3. Farming Chatbot: rule-based (keyword matching of plant, disease and intent: prevent, symptoms, cause,
   spread, treat, chemicals). It remembers the last diagnosed disease for follow-up questions.
   It never recommends specific pesticides or dosages (safety by design).
Knowledge base: 26 disease entries (cause, symptoms, spread) + care/prevention by disease type.
Note for references: the knowledge base is compiled from general plant pathology knowledge.
The content has not been reviewed by an agricultural expert (this is stated as a limitation in the report).

Error handling: no image uploaded, missing model weights (clear message), unreadable image, unknown question.

## 7. Code structure (repository folder hos052)
- src/app.py - Gradio user interface (3 tabs)
- src/classifier.py - model loading, preprocessing, prediction (shared class list for both models)
- src/explain.py - Grad-CAM heatmaps
- src/knowledge_base.py - disease data, report builder, chatbot logic
- notebooks/train_agroguard.ipynb - Colab training notebook (all steps: datasets, class mapping, baseline evaluation, training of Model 2 and 3, plots)
- docs/results.md, docs/project_log.md - results and notes
- models/ (ignored by git) - local weights models/model3 (config.json, model.safetensors, about 9 MB). Public download link (Google Drive): https://drive.google.com/drive/folders/1UUDsUV2dfipey_0GkEDTA39S0c2IYRLv?usp=sharing
- data/ (ignored by git) - datasets are downloaded by the notebook; links are given in the README
- requirements.txt, README.md, .gitignore
Weights and datasets are not uploaded to GitHub (assignment rule). Model 3 is shared through a Google Drive link
given in the README.

## 8. Git / GitHub workflow
- Cloned xahiru/PT22026, created branch hos052-project, folder hos052 provided by the TAs.
- Direct push to xahiru/PT22026 failed with 403 (no write permission), so the TA instructed:
  fork the repository, work in the hos052 folder of the fork, then open a Pull Request to the main repo.
- Fork: https://github.com/hossainmdpial52/PT22026 (remote name "myfork"); branch hos052-project pushed.
- Commits so far: initial project structure; basic Gradio app with pre-trained model;
  modular classifier + Model 3 + comparison dropdown + results notes; knowledge base, Disease Library tab and
  Farming Chatbot; training notebook; README and requirements; Grad-CAM tab and README update.
- Git push needed a proxy setting on the student computer (git config --global http.proxy http://127.0.0.1:12000).
- The Pull Request is the official submission and is created only when everything is finished.

## 9. Tools and environment
Python 3.12, PyTorch, torchvision, Hugging Face Transformers and Datasets, Gradio 6,
scikit-learn, grad-cam (installed), pandas, matplotlib. Training on Google Colab (Tesla T4).
Checkpoints, final models, results.json and plots are stored in Google Drive folder hos052_agroguard.

## 10. Status
All planned features are finished: Diagnose, Disease Library, Farming Chatbot and Explain (Grad-CAM) tabs,
training notebook, README, requirements.txt and .gitignore. Model weights are shared through Google Drive
(link in the README). The technical report is written separately as a PDF.

## 11. Figures in the report
1 PlantVillage samples, 2 Diagnose tab, 3 Disease Library tab, 4 Farming Chatbot tab, 5 Grad-CAM tab,
6 GitHub folder page, 7 Pull Request page, 8 model comparison chart, 9 training curves, 10 confusion matrix,
11 Model 3 vs ready-made baseline on the same field photo, 12 ready-made baseline on a lab-style photo.

## 12. Grad-CAM observations (for the Discussion section)
- Test photo: apple leaf with many red-brown spots. Model 3 predicted Apple - Cedar apple rust (76% confidence).
  The heatmap's red area was on the left-center part of the leaf, where several large lesions are, not on the background.
- The heatmap did not mark every lesion (some lesions on the right side stayed blue). Reason: Grad-CAM works on a
  7 x 7 feature map, so it is coarse; it shows where the model looked, not a proof that the answer is right.
- How Grad-CAM works: gradients of the class score w.r.t. the last conv feature maps are averaged per channel (weights),
  the weighted feature maps are summed, ReLU is applied, the map is resized to 224 x 224 and overlaid on the photo.
- Reference: Selvaraju et al., ICCV 2017 (arXiv:1610.02391); library: jacobgil/pytorch-grad-cam.
