import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images")

# --- Model Settings ---
# CLIP model that maps images and text into the same vector space
CLIP_MODEL_NAME = 'clip-ViT-B-32'