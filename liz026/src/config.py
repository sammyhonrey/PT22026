
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_FILENAME = "pneumonia_detection_Vision_Model.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
REPORT_DIR = PROJECT_ROOT / "reports"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

MODEL_URL = (
    "https://github.com/Youssefx64/Xray-Analysis-using-CNN-and-gradio-deployment"
    "/raw/main/models/pneumonia_detection_Vision_Model.keras"
)

# ---------------------------------------------------------------- model
IMAGE_SIZE = (224, 224)            # MobileNetV2 input resolution
CLASS_NAMES = ("Normal", "Pneumonia")
DEFAULT_THRESHOLD = 0.5            # sigmoid output >= threshold -> Pneumonia
UNCERTAIN_LOGIT_MARGIN = 0.6         # "close to the threshold" band (0.35-0.65 at t = 0.5)
BACKBONE_LAYER = "mobilenetv2_1.00_224"   # nested MobileNetV2 sub-model
LAST_CONV_LAYER = "out_relu"              # last conv activation (7x7x1280), used by Grad-CAM

# ---------------------------------------------------------------- input checks
MIN_IMAGE_SIDE = 64                # reject tiny images
MAX_COLOR_DEVIATION = 25.0         # mean |R-G|,|G-B| above this -> probably not an X-ray

# ---------------------------------------------------------------- server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
