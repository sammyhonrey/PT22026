"""Central settings shared by training, evaluation and the app."""
IMG_SIZE = 64                      # every image is resized to 64x64
MEAN = (0.3403, 0.3121, 0.3214)    # approximate GTSRB channel statistics
STD = (0.2724, 0.2608, 0.2669)
MODEL_NAMES = ("smallcnn", "resnet18", "convnext_tiny")

# Weather / image-quality conditions used for the robustness study.
CONDITIONS = ("fog", "night", "rain", "blur", "noise", "glare")
SEVERITIES = (1, 2, 3, 4, 5)

# Weather augmentation during training uses severities 1-4 and every condition EXCEPT
# "glare". Severity 5 and glare are therefore never seen in training, so evaluating on them
# measures generalisation to harder / unseen conditions rather than memorisation.
AUG_CONDITIONS = ("fog", "night", "rain", "blur", "noise")
AUG_MAX_SEVERITY = 4

DEFAULT_REJECT_THRESHOLD = 0.70    # below this confidence the app says "not sure"
SEED = 42
