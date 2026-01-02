import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image, ImageFilter
import matplotlib.cm as cm
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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


### Reall part 

def load_model():
    model = models.resnet50(pretrained=False)

    # ⚠️ IMPORTANT: must match training
    num_classes = 5   # Healthy, Condition A, Condition B
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(
        "resnet50_weight_best_model.pth",
        map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    model.eval()
    return model

def predict(image, model):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()

    classes = ["Healthy", "Condition A", "Condition B", "Condition C", "Condition D"]
    pred_idx = int(np.argmax(probs))

    return {
        "class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probs": dict(zip(classes, probs))
    }
    
def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs






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

    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=None
    )[0]

    visualization = show_cam_on_image(
        original_image,
        grayscale_cam,
        use_rgb=True
    )

    return visualization











def gradcam_explainability(model, img_tensor, original_image):
    """
    model: PyTorch model
    img_tensor: torch.Tensor [1, 3, H, W] (prétraitée)
    original_image: PIL Image ou numpy array [H, W, 3] (0-1)
    """

    model.eval()

    # Last conv layer
    target_layers = [model.layer4[-1]]

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Grad-CAM heatmap (224x224)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    heatmap = grayscale_cam[0]  # shape: (224,224)

    # Convert original image to numpy float32 0-1
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image).astype(np.float32) / 255.0
    else:
        original_np = original_image.astype(np.float32)

    H, W, _ = original_np.shape

    # Resize heatmap to match original image size
    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255))   # heatmap -> PIL
    heatmap_resized = heatmap_pil.resize((W, H), resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized).astype(np.float32) / 255.0

    # Convert grayscale to 3 channels
    heatmap_resized = np.stack([heatmap_resized]*3, axis=-1)  # (H,W,3)

    # Overlay
    visualization = original_np * 0.6 + heatmap_resized * 0.4
    visualization = np.clip(visualization, 0, 1)

    return visualization