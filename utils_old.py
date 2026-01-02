import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# =========================
# Constants
# =========================
CLASSES = [
    "Healthy",
    "Class 1",
    "Class 2",
    "Class 3",
    "Class 4"
]

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
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# =========================
# Load trained model
# =========================
def load_model():
    model = models.resnet50(pretrained=False)

    num_classes = len(CLASSES)  # MUST match training
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(
        "resnet50_weight_best_model.pth",
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model

# =========================
# Prediction
# =========================
def predict(model, input_tensor):
    """
    model: PyTorch model
    input_tensor: torch.Tensor [1, 3, 224, 224]
    """

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # shape: [num_classes]

    pred_idx = torch.argmax(probs).item()

    return {
        "class": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": dict(zip(CLASSES, probs.cpu().numpy()))
    }

# =========================
# Grad-CAM Explainability
# =========================

def gradcam_explainability(model, img_tensor, original_image):
    """
    model: PyTorch model
    img_tensor: torch.Tensor [1, 3, H, W]
    original_image: numpy array [H, W, 3] in range [0, 1]
    """
    
    model.eval()
    
    target_layers = [model.layer4[-1]]
    
    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )
    
    # Generate Grad-CAM heatmap
    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=None
    )[0, :]
    
    # RESIZE heatmap using PyTorch (more accurate than simple NumPy)
    # Convert to torch tensor and add batch and channel dimensions
    heatmap_tensor = torch.tensor(grayscale_cam).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Resize using PyTorch interpolation
    target_height, target_width = original_image.shape[:2]
    heatmap_resized_tensor = F.interpolate(
        heatmap_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to numpy
    heatmap_resized = heatmap_resized_tensor.squeeze().numpy()
    
    # Apply heatmap to image
    visualization = show_cam_on_image(
        original_image,
        heatmap_resized,
        use_rgb=True
    )
    
    return visualization
