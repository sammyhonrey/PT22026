"""Fine-tune YOLOv8n for helmet detection (With Helmet / Without Helmet).

This script mirrors the training that produced models/best.pt.
It was run on Google Colab (free T4 GPU), because a GPU is required
for reasonable training time.

Usage (Colab or any machine with a GPU):
    export ROBOFLOW_API_KEY="your_key"      # never hard-code the key
    python train.py --epochs 50 --imgsz 640 --batch 16
"""
import argparse
import os

from roboflow import Roboflow
from ultralytics import YOLO

# Dataset: Bike Helmet Detection (Roboflow Universe, Public Domain)
WORKSPACE = "bike-helmets"
PROJECT = "bike-helmet-detection-2vdjo"
VERSION = 2


def get_api_key() -> str:
    """Read the Roboflow API key from the environment or Colab secrets."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    try:  # Google Colab: Secrets panel
        from google.colab import userdata
        return userdata.get("ROBOFLOW_API_KEY")
    except Exception as exc:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not found. Set it as an environment variable "
            "or add it to Colab Secrets."
        ) from exc


def download_dataset() -> str:
    """Download the dataset in YOLOv8 format and return its folder path."""
    rf = Roboflow(api_key=get_api_key())
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset.location


def train(data_yaml: str, args) -> str:
    """Fine-tune the pre-trained YOLOv8n model and return the best weights path."""
    model = YOLO("yolov8n.pt")  # small COCO-pretrained model
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        save_period=5,
        exist_ok=True,
    )
    return os.path.join(args.project, args.name, "weights", "best.pt")


def evaluate(weights: str, data_yaml: str) -> None:
    """Evaluate the trained model on the held-out test split."""
    metrics = YOLO(weights).val(data=data_yaml, split="test")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n for helmet detection")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs", help="output folder")
    parser.add_argument("--name", default="helmet_yolov8n", help="run name")
    args = parser.parse_args()

    location = download_dataset()
    data_yaml = os.path.join(location, "data.yaml")
    best = train(data_yaml, args)
    evaluate(best, data_yaml)


if __name__ == "__main__":
    main()

