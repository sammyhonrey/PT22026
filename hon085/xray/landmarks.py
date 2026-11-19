"""Names and groupings of the 33 BlazePose landmarks."""

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
INDEX = {name: i for i, name in enumerate(NAMES)}
COUNT = len(NAMES)

BODY = [INDEX[n] for n in (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
)]

MIRROR = list(range(COUNT))
for _name, _i in INDEX.items():
    if _name.startswith("left_"):
        _j = INDEX["right_" + _name[5:]]
        MIRROR[_i], MIRROR[_j] = _j, _i
    elif _name.endswith("_left"):
        _j = INDEX[_name[:-5] + "_right"]
        MIRROR[_i], MIRROR[_j] = _j, _i

ANGLES = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
}

DISTANCES = [
    ("left_wrist", "right_wrist"), ("left_ankle", "right_ankle"),
    ("left_elbow", "right_elbow"), ("left_knee", "right_knee"),
    ("left_wrist", "left_ankle"), ("right_wrist", "right_ankle"),
    ("left_wrist", "left_hip"), ("right_wrist", "right_hip"),
    ("left_ankle", "left_hip"), ("right_ankle", "right_hip"),
    ("left_wrist", "nose"), ("right_wrist", "nose"),
    ("left_ankle", "right_knee"), ("right_ankle", "left_knee"),
]
