"""
Loading the pretrained CNN.

The model is a MobileNetV2 backbone (ImageNet weights) with a small dense
classification head, fine-tuned for NORMAL vs PNEUMONIA chest X-rays.
"""
import logging
import urllib.request

from src import config  # sets KERAS_BACKEND before keras is imported
import keras

logger = logging.getLogger(__name__)

_model = None  # cached instance (loaded once per process)


def download_model(url: str = config.MODEL_URL, dest=config.MODEL_PATH) -> None:
    """Download the pretrained .keras file if it is not on disk yet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    logger.info("Downloading model from %s ...", url)
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download the model ({exc}). Download it manually from\n"
            f"  {url}\nand save it as\n  {dest}"
        ) from exc
    logger.info("Model saved to %s", dest)


def get_model():
    """Return the loaded Keras model, downloading/loading it on first call."""
    global _model
    if _model is not None:
        return _model

    if not config.MODEL_PATH.exists():
        download_model()

    try:
        # compile=False: we only need inference, not the training optimizer.
        _model = keras.models.load_model(config.MODEL_PATH, compile=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {config.MODEL_PATH}: {exc}") from exc

    logger.info("Model loaded (%s backend, %s parameters)",
                keras.backend.backend(), f"{_model.count_params():,}")
    return _model
