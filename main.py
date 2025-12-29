import streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=img_tensor)
heatmap = grayscale_cam[0]

visualization = show_cam_on_image(
    original_image,
    heatmap,
    use_rgb=True
)
