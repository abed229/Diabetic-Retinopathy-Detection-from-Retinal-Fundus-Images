import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# =========================
# Image preprocessing
# =========================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)


# =========================
# Fake model prediction
# =========================
def fake_predict():
    classes = ["Healthy", "Condition A", "Condition B"]
    probs = np.random.dirichlet(np.ones(len(classes)))
    pred_idx = np.argmax(probs)

    return {
        "class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": dict(zip(classes, probs))
    }


# =========================
# Fake explainability (heatmap)
# =========================
def fake_explainability(image):
    image = np.array(image).astype(np.float32) / 255.0
    h, w, _ = image.shape

    heatmap = np.random.rand(h, w)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )

    overlay = 0.6 * image + 0.4 * (heatmap / 255.0)
    return overlay
