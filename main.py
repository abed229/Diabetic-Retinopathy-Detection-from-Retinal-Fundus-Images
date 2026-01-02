import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        output[:, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

def overlay_gradcam(cam, original_image):
    """
    original_image: numpy array in [0,1], shape (H,W,3)
    """
    h, w, _ = original_image.shape

    # Resize CAM
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    cam_img = np.array(cam_img) / 255.0

    # Apply colormap
    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam_img)[:, :, :3]  # drop alpha

    # Overlay
    overlay = 0.4 * heatmap + 0.6 * original_image
    overlay = np.clip(overlay, 0, 1)

    return overlay


def overlay_gradcam(cam, original_image):
    """
    original_image: numpy array in [0,1], shape (H,W,3)
    """
    h, w, _ = original_image.shape

    # Resize CAM
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    cam_img = np.array(cam_img) / 255.0

    # Apply colormap
    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam_img)[:, :, :3]  # drop alpha

    # Overlay
    overlay = 0.4 * heatmap + 0.6 * original_image
    overlay = np.clip(overlay, 0, 1)

    return overlay

def gradcam_explainability(model, input_tensor, original_image):
    target_layer = model.layer4[-1].conv3  # ResNet50

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    cam_image = overlay_gradcam(cam, original_image)
    return cam_image