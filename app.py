import streamlit as st
from PIL import Image
import numpy as np

from utils import preprocess_image, fake_predict, fake_explainability

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="EyeScan AI",
    page_icon="👁️",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.metric-card {
    background-color: #f7f9fc;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Settings")
use_camera = st.sidebar.checkbox("Use camera", value=True)
show_explain = st.sidebar.checkbox("Show explainability", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ This application is for **research and educational purposes only**.\n\n"
    "It does NOT provide medical diagnosis."
)

# =========================
# Main title
# =========================
st.title("👁️ EyeScan AI")
st.markdown(
    "AI-assisted eye image analysis with **explainable predictions**."
)

# =========================
# Image input
# =========================
image = None

if use_camera:
    camera_input = st.camera_input("Take a photo of the eye")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")
else:
    uploaded = st.file_uploader(
        "Upload an eye image",
        type=["jpg", "png", "jpeg"]
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")

# =========================
# Prediction pipeline
# =========================
if image:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        st.image(image, use_container_width=True, clamp=True)

    with col2:
        st.subheader("🧠 Model Prediction")

        with st.spinner("Analyzing image..."):
            _ = preprocess_image(image)  # ready for real model
            result = fake_predict()

        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{result['class']}</h3>
                <p>Confidence</p>
                <h2>{result['confidence']*100:.1f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 📊 Class probabilities")
        for k, v in result["all_probs"].items():
            st.progress(float(v), text=f"{k} ({v*100:.1f}%)")

    # =========================
    # Explainability
    # =========================
    if show_explain:
        st.markdown("---")
        st.subheader("🔥 Explainability (visual attention)")

        heatmap_img = fake_explainability(image)
        st.image(
            heatmap_img,
            caption="Highlighted regions influencing the prediction",
            use_container_width=True
        )

else:
    st.info("Please upload or capture an eye image to start analysis.")
