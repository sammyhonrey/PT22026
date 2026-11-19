
"""Configuration for the SceneSense segmentation application."""

# Pretrained model used for semantic segmentation.
MODEL_ID = "facebook/maskformer-resnet50-coco-stuff"

# Pretrained model used for instance and panoptic segmentation.
INSTANCE_MODEL_ID = "facebook/maskformer-swin-tiny-coco"

DEVICE = "cpu"
NUM_CLASSES = 171

# The first 80 COCO categories represent countable objects.
INSTANCE_THING_CLASSES = 80

INSTANCE_SCORE_THRESHOLD = 0.5

# Ignore very small predictions in instance segmentation.
MIN_INSTANCE_AREA_PERCENT = 0.30

DEFAULT_OVERLAY_OPACITY = 0.35

MAX_IMAGE_SIDE = 640
MAX_INPUT_PIXELS = 40_000_000
MIN_IMAGE_SIDE = 32
