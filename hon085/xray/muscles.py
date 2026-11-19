"""Muscle groups of the anatomy model and the groups each pose works."""

import re

GROUPS = {
    "quadriceps": r"rectus femoris|vastus",
    "glutes": r"gluteus",
    "hamstrings": r"biceps femoris|semitendinosus|semimembranosus",
    "calves": r"gastrocnemius|soleus",
    "adductors": r"adductor (magnus|longus|brevis|minimus)|gracilis|pectineus",
    "hip_flexors": r"psoas|iliacus",
    "abdominals": r"rectus abdominis|(external|internal) oblique|transversus abdominis",
    "back_extensors": r"iliocostalis|longissimus|spinalis|multifidus|quadratus lumborum",
    "upper_back": r"trapezius|rhomboid",
    "lats": r"latissimus",
    "serratus": r"serratus anterior",
    "chest": r"pectoralis major",
    "shoulders": r"deltoid",
    "biceps": r"biceps brachii|brachialis",
    "triceps": r"triceps brachii",
    "forearms": r"carpi|brachioradialis|palmaris|pronator",
}

GROUP_TITLES = {
    "quadriceps": "Quadriceps", "glutes": "Gluteal muscles", "hamstrings": "Hamstrings",
    "calves": "Calf muscles", "adductors": "Hip adductors", "hip_flexors": "Hip flexors",
    "abdominals": "Abdominal muscles", "back_extensors": "Spinal extensors",
    "upper_back": "Trapezius and rhomboids", "lats": "Latissimus dorsi",
    "serratus": "Serratus anterior", "chest": "Pectoralis major", "shoulders": "Deltoids",
    "biceps": "Elbow flexors", "triceps": "Triceps", "forearms": "Forearm muscles",
}

PRIMARY, SECONDARY = 1.0, 0.5

POSES = {
    "standing": {
        "title": "Standing",
        "primary": [],
        "secondary": ["back_extensors", "calves"],
        "cue": "Upright stance. Only the postural muscles are lightly active.",
    },
    "squat": {
        "title": "Bodyweight squat",
        "primary": ["quadriceps", "glutes"],
        "secondary": ["hamstrings", "adductors", "back_extensors", "abdominals", "calves"],
        "cue": "Hips back and down, chest up, knees in line with the toes.",
    },
    "chair": {
        "title": "Chair pose (Utkatasana)",
        "primary": ["quadriceps", "glutes", "back_extensors"],
        "secondary": ["shoulders", "calves", "abdominals", "adductors"],
        "cue": "Knees bent as if sitting back on a chair, arms reaching up.",
    },
    "tree": {
        "title": "Tree pose (Vrksasana)",
        "primary": ["glutes", "quadriceps", "calves"],
        "secondary": ["abdominals", "adductors", "shoulders"],
        "cue": "Stand on one leg, the other foot against the inner thigh, hands together.",
    },
    "warrior": {
        "title": "Warrior III (Virabhadrasana III)",
        "primary": ["glutes", "hamstrings", "back_extensors"],
        "secondary": ["quadriceps", "calves", "abdominals", "shoulders"],
        "cue": "Balance on one leg with the body and the other leg in one horizontal line.",
    },
    "cobra": {
        "title": "Cobra pose (Bhujangasana)",
        "primary": ["back_extensors", "glutes", "triceps"],
        "secondary": ["upper_back", "shoulders", "hamstrings"],
        "cue": "Lying face down, chest lifted, shoulders away from the ears.",
    },
    "dog": {
        "title": "Downward dog (Adho Mukha Svanasana)",
        "primary": ["shoulders", "triceps", "serratus"],
        "secondary": ["upper_back", "lats", "quadriceps", "abdominals"],
        "cue": "Hands and feet on the floor, hips high, back long.",
    },
}

_COMPILED = {name: re.compile(pattern) for name, pattern in GROUPS.items()}


def group_of(part_name):
    """Muscle group of a BodyParts3D part name, or None."""
    name = part_name.lower()
    for group, pattern in _COMPILED.items():
        if pattern.search(name):
            return group
    return None


def activation(pose, extra=None):
    """Map of muscle group to intensity (0..1) for a pose label."""
    info = POSES.get(pose) or (extra or {}).get(pose)
    if not info:
        return {}
    levels = {g: SECONDARY for g in info.get("secondary", []) if g in GROUPS}
    levels.update({g: PRIMARY for g in info.get("primary", []) if g in GROUPS})
    return levels


def describe(pose, extra=None):
    info = POSES.get(pose) or (extra or {}).get(pose) or {}
    return {
        "title": info.get("title", pose.replace("_", " ").title()),
        "cue": info.get("cue", ""),
        "primary": [GROUP_TITLES[g] for g in info.get("primary", []) if g in GROUPS],
        "secondary": [GROUP_TITLES[g] for g in info.get("secondary", []) if g in GROUPS],
    }
