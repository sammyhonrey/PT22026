# Dataset Notes

## Main dataset
- **Name:** Garbage Classification (12 classes)
- **Source:** https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **Owner:** Kaggle user `mostafaabla` (the dataset page lists no author names)
- **Version / date:** Version 1; the downloaded file is dated 24 January 2021
- **Licence:** "Database: Open Database, Contents: (c) Original Authors" as shown on the Kaggle page; the Kaggle tool reports the database licence as ODbL-1.0
- **Size:** 250.51 MB unpacked (239 MB ZIP), 15,515 images, all `.jpg`, in 12 class folders
- **Original classes and counts:** battery 945, biological 985, brown-glass 607, cardboard 891, clothes 5325, green-glass 629, metal 769, paper 1050, plastic 865, shoes 1977, trash 697, white-glass 775
- **Use in this project:** academic coursework only. The images are not redistributed: they stay on Google Drive and Colab and are never committed to GitHub.

## Where the data lives
- ZIP on Google Drive: `SmartSort_AI/data/garbage-classification.zip`
- Unzipped on Colab local disk for training: `/content/data/garbage_classification/`

## Class merge
brown-glass, green-glass and white-glass were merged into one class `glass` (2,011 images),
giving 10 classes: battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash
(15,515 images in total). Glass colour does not change which bin the item goes in, so this removes
a distinction the application does not need. The raw folders are left untouched; the merge is
done by label mapping when the split file is built.

## Known issues
- Strong class imbalance: clothes (5,325) is about 7.6 times larger than trash (697).
- The Kaggle page does not name the original image authors or describe how the images were collected.
- Duplicates, corrupt files and possible label noise have not been checked yet; this is done during
  data cleaning and exploratory analysis.

## Related dataset (not trained on)
- **TrashNet** (Yang and Thung, Stanford CS 229 course project): 2,527 images in 6 classes
  (glass 501, paper 594, cardboard 403, plastic 482, metal 410, trash 137), photographed on a white
  poster board and resized to 512 x 384.
- **Source:** https://github.com/garythung/trashnet (data also on Hugging Face: `garythung/trashnet`)
- **Licence:** MIT, according to the repository metadata (re-check the repository's LICENSE file
  before submission). The authors ask users of the dataset to cite their repository.
- Discussed in Related Work; not used for training because it has no clothes, shoes, battery or
  biological classes.

## Real-photo test set
- 214 photos taken with my own phone (Samsung Galaxy S24), 19 to 28 per class, in 10 class folders:
  battery 21, biological 19, cardboard 21, clothes 19, glass 21, metal 19, paper 19, plastic 24, shoes 23, trash 28.
- The trash class was extended with 8 photos of items of the kind the dataset calls trash (face masks, toothbrushes and similar hygiene items). This was done after looking at the dataset's trash class and before any model was evaluated on real photos.
- Taken with varied backgrounds, lighting and angles, with at least 5 different items per class.
- Photos contain no faces or personal information; location metadata was removed.
- Resized to a maximum of 1024 px on the long side with `src/resize_photos.py`.
- Stored on Google Drive (`SmartSort_AI/real_photos`), not in Git.
- Never used for training or tuning; used once at the end to measure the real-world gap.

## Cleaning and splitting
- **Scan:** all 15,515 images were opened and verified: 0 unreadable files. 34 images are palette
  mode and are converted to RGB when loaded.
- **Exact duplicates** (identical bytes, MD5): 37 files in 18 groups. 22 files were removed:
  16 plain copies, and 6 files that are three images appearing under two different labels
  (white-glass and plastic or metal), which were removed entirely. 15,493 images remain.
  Details are in `splits/cleaning_log.csv`.
- **Near-duplicates:** a perceptual hash (pHash, Hamming distance of 4 or less) gave 232 candidate
  pairs. pHash alone also matched unrelated items on white backgrounds, so each candidate was
  verified by the correlation of 32 x 32 grayscale thumbnails (0.90 or higher). 128 pairs were
  verified, forming 113 groups (largest: 4 images), none mixing labels. All images of a group
  are always placed in the same split.
- **Split:** stratified by class at group level, seed 42, 70/15/15, saved as `splits/split_v1.csv`
  (columns: path, label, split, orig_class): train 10,847, validation 2,323, test 2,323.
- **Observations:** the dataset contains recoloured copies of the same photo (for example brown
  and green glass), transparent glass and plastic are sometimes ambiguous, and the `trash`
  class consists of toothbrushes, face masks and diapers.