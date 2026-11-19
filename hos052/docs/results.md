# Experiment Results (AgroGuard AI)

## Datasets
- PlantVillage (Hugging Face: BrandonFors/Plant-Diseases-PlantVillage-Dataset): 43,456 train / 10,849 test images, 38 classes. Lab images with plain background.
- PlantDoc (GitHub: pratikkayal/PlantDoc-Dataset, CC BY 4.0): 2,342 train / 236 test field images, 27 classes mapped to PlantVillage classes. The class "Tomato two spotted spider mites leaf" (2 images) is excluded.

## Training setup
- Architecture: MobileNetV2 (ImageNet pre-trained, google/mobilenet_v2_1.0_224), new 38-class head
- Epochs: 12, batch size 64, optimizer AdamW (lr 5e-4, weight decay 1e-4), cosine learning-rate schedule
- Loss: cross-entropy with label smoothing 0.1, mixed precision (fp16), GPU: Tesla T4 (Google Colab)
- Augmentation: random resized crop (0.4-1.0), horizontal/vertical flip, rotation 30 degrees, color jitter
- Model 3 training data: PlantVillage train + PlantDoc train repeated 6 times
- The reported numbers are from the final (12th) epoch, not the best epoch.

## Test accuracy

| Model | Lab images (PlantVillage test) | Field images (PlantDoc test) |
|---|---|---|
| Ready-made baseline | 95.58% | 31.78% |
| Model 2 (PlantVillage only) | 99.33% | 27.54% |
| Model 3 (PlantVillage + PlantDoc) | 99.27% | 67.37% |

## Notes
- The PlantDoc test set has only 236 images, so one image is about 0.42%. Small differences are not reliable.
- Weak classes of Model 3 on field images: Corn Northern Leaf Blight vs Gray leaf spot, Potato early vs late blight, Tomato healthy, Tomato mosaic virus, Tomato bacterial spot.
