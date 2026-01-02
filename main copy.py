import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def gradcam_explainability(model, img_tensor, original_image):
    """
    model: PyTorch model
    img_tensor: torch.Tensor [1, 3, H, W]
    original_image: numpy array [H, W, 3] in range [0,1]
    """

    model.eval()

    # Target layer: last conv layer of ResNet
    target_layers = [model.layer4[-1]]

    # Grad-CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=torch.cuda.is_available()
    )

    grayscale_cam = cam(
        input_tensor=img_tensor,
        targets=None
    )

    heatmap = grayscale_cam[0]

    # Overlay heatmap on image
    visualization = show_cam_on_image(
        original_image,
        heatmap,
        use_rgb=True
    )

    return visualization
