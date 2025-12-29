import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.cm as cm

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
    # Convert image to numpy
    image_np = np.array(image).astype(np.float32) / 255.0
    h, w, _ = image_np.shape

    # Fake attention map
    heatmap = np.random.rand(h, w).astype(np.float32)

    # Gaussian blur using PIL (no OpenCV)
    heatmap_img = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=25))
    heatmap = np.array(heatmap_img) / 255.0

    # Apply colormap using matplotlib
    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap)[:, :, :3]

    # Overlay
    overlay = 0.6 * image_np + 0.4 * heatmap_color
    overlay = np.clip(overlay, 0, 1)

    return overlay
