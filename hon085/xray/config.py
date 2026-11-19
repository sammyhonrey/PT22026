from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TASK_DIR = WEB_DIR / "models"

RAW_DIR = DATA_DIR / "raw"
LANDMARK_CSV = DATA_DIR / "pose_landmarks.csv"
USER_CSV = DATA_DIR / "user_samples.csv"
CLASSIFIER_PATH = MODEL_DIR / "pose_classifier.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

YOGA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/pose_classification/yoga_poses.zip"
SQUAT_URL = "https://raw.githubusercontent.com/nihar787airobotics/squat_dataset_publication/main/data/raw/{view}/{name}.mp4"
SQUAT_PEOPLE = 20
SQUAT_TEST_PEOPLE = (17, 18, 19, 20)
YOGA_DIR = RAW_DIR / "yoga_poses"
SQUAT_DIR = RAW_DIR / "squat_videos"
TASK_URL = "https://storage.googleapis.com/mediapipe-models/{kind}/{name}/float16/latest/{name}.task"
TASK_FILES = {
    "pose_landmarker_lite": "pose_landmarker",
    "pose_landmarker_full": "pose_landmarker",
    "pose_landmarker_heavy": "pose_landmarker",
    "hand_landmarker": "hand_landmarker",
}

HOST = "127.0.0.1"
PORT = 8000
MIN_VISIBILITY = 0.5
UNKNOWN = "unknown"
