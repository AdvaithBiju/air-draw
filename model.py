"""Pretrained Quick, Draw! classifier (100 classes, 28x28 input) from
https://huggingface.co/nateraw/quickdraw (MIT license)."""
import os

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://huggingface.co/nateraw/quickdraw/resolve/main/pytorch_model.bin"
LABELS_URL = "https://huggingface.co/nateraw/quickdraw/resolve/main/class_names.txt"
MODEL_PATH = os.path.join(HERE, "quickdraw_pretrained.bin")
LABELS_PATH = os.path.join(HERE, "class_names.txt")

# What the game asks you to draw (subset of the 100 classes; plain shapes like "circle" left out)
TARGETS = [
    "cat", "apple", "bicycle", "bird", "butterfly", "car", "clock", "cloud", "cup",
    "donut", "eye", "flower", "hat", "ice_cream", "key", "ladder", "light_bulb",
    "lightning", "moon", "mountain", "mushroom", "pizza", "rainbow", "smiley_face",
    "snake", "spider", "star", "sun", "sword", "tent", "tree", "umbrella",
    "airplane", "candle", "envelope", "t-shirt", "eyeglasses", "headphones",
]


def _download(url, path):
    if not os.path.exists(path):
        import urllib.request
        print("downloading", url)
        urllib.request.urlretrieve(url, path)


def load_model():
    """Returns (model, labels). Downloads the weights on first run."""
    _download(MODEL_URL, MODEL_PATH)
    _download(LABELS_URL, LABELS_PATH)
    with open(LABELS_PATH) as f:
        labels = f.read().split()
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding="same"), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1152, 256), nn.ReLU(),
        nn.Linear(256, len(labels)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, labels


def pretty(label):
    return label.replace("_", " ")
