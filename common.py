import os

CLASSES_FRAMEWORK = {
    "clean": 0,
    "cadv": 1,
    "cnn-attack": 2,
    "grad-est": 3,
    "ifgsm": 4,
    "korhonen-et-al": 5,
    "madc": 6,
    "onepixel": 7,
    "parsimonious": 8,
    "patch-rs": 9,
    "square-attack": 10,
    "ssah": 11,
    "uap": 12,
    "zhang-et-al-dists": 13,
}

METRICS = {
    "clean": 0,
    "clip-iqa-plus": 1,
    "fpr": 2,
    "koncept": 3,
    "linearity": 4,
    "maniqa": 5,
    "meta-iqa": 6,
    "paq2piq": 7,
    "spaq": 8,
    "topiq-nr": 9,
}

if os.path.exists("/scratch/pr0207/users/27a_leb"):
    DATASET_PATH = "/scratch/pr0207/users/27a_leb"
    MODEL_PATH = "/scratch/pr0207/users/27a_leb/attack-detection/models"
else:
    DATASET_PATH = "../"
    MODEL_PATH = "models"

MODEL_PATH_FRAMEWORK = MODEL_PATH
