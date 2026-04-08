import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Low-Light Unsafe Zone Detection", layout="wide")

label_map = {
    0: "Car",
    1: "Bus",
    2: "Bicycle",
    3: "Motorbike"
}

@st.cache_resource
def load_model():
    model_path = "unsafe_zone_cnn_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model()

def gamma_correction(image, gamma=1.8):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def denoise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def final_optimize_soft(image):
    gamma_img = gamma_correction(image, 1.8)
    clahe_img = apply_clahe(gamma_img)
    enhanced_img = denoise(clahe_img)
    return enhanced_img

def kmeans_segmentation(image, k=5):
    pixel_vals = image.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)

    return segmented_image

def generate_unsafe_mask(enhanced):
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    unsafe_mask = (thresh > 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    unsafe_mask = cv2.morphologyEx(unsafe_mask, cv2.MORPH_OPEN, kernel)
    unsafe_mask = cv2.morphologyEx(unsafe_mask, cv2.MORPH_DILATE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(unsafe_mask, connectivity=8)
    filtered_mask = np.zeros_like(unsafe_mask)
    min_area = 100

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            filtered_mask[labels == i] = 1

    unsafe_mask = filtered_mask

    height = gray.shape[0]
    roi_mask = np.zeros_like(gray)
    roi_mask[int(height * 0.4):, :] = 1
    unsafe_mask = unsafe_mask * roi_mask

    return unsafe_mask, gray

def mask_to_display(unsafe_mask):
    return (unsafe_mask * 255).astype(np.uint8)

def create_overlay(original, unsafe_mask):
    overlay = original.copy()
    red_layer = np.zeros_like(original)
    red_layer[:, :, 0] = 255
    alpha = 0.35

    overlay[unsafe_mask == 1] = (
        alpha * red_layer[unsafe_mask == 1] +
        (1 - alpha) * original[unsafe_mask == 1]
    ).astype(np.uint8)

    return overlay

def create_heatmap(original, unsafe_mask):
    heatmap_input = (unsafe_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap_input, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay_heatmap = cv2.addWeighted(original, 0.65, heatmap_rgb, 0.35, 0)
    return overlay_heatmap

def compute_safety_score(gray, unsafe_mask):
    unsafe_ratio = np.sum(unsafe_mask) / unsafe_mask.size
    avg_brightness = np.mean(gray)

    brightness_score = (avg_brightness / 255.0) * 100
    unsafe_penalty = unsafe_ratio * 100

    safety_score = max(0, min(100, 0.6 * brightness_score + 0.4 * (100 - unsafe_penalty)))

    if safety_score >= 70:
        risk_level = "SAFE"
        recommendation = "Pedestrian zone visibility is within acceptable limits."
    elif safety_score >= 40:
        risk_level = "MODERATE RISK"
        recommendation = "Moderate visibility risk detected. Additional illumination is advised."
    else:
        risk_level = "HIGH RISK"
        recommendation = "Immediate lighting improvement and pedestrian warning recommended."

    return safety_score, risk_level, recommendation

def cnn_predict(image_rgb):
    cnn_input = cv2.resize(image_rgb, (128, 128))
    cnn_input = cnn_input / 255.0
    cnn_input = np.expand_dims(cnn_input, axis=0)

    prediction = model.predict(cnn_input, verbose=0)[0]
    pred_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return label_map[pred_index], confidence

def final_display_image(image):
    enhanced = final_optimize_soft(image)

    # slight brightness/contrast adjustment
    final_img = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=10)

    # smooth noise while preserving edges
    final_img = cv2.fastNlMeansDenoisingColored(final_img, None, 6, 6, 7, 21)

    return final_img

st.title("Low-Light Image Enhancement and Unsafe Zone Analysis")
st.write("Upload a low-light traffic image to run enhancement, segmentation, unsafe-region analysis, and CNN-based classification.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = np.array(pil_img)

    enhanced = final_optimize_soft(img)
    final_output = final_display_image(img)
    segmented_img = kmeans_segmentation(enhanced, k=5)
    unsafe_mask, gray = generate_unsafe_mask(enhanced)
    mask_display = mask_to_display(unsafe_mask)
    overlay = create_overlay(img, unsafe_mask)
    heatmap = create_heatmap(img, unsafe_mask)
    safety_score, risk_level, recommendation = compute_safety_score(gray, unsafe_mask)
    pred_label, confidence = cnn_predict(enhanced)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(enhanced, caption="Enhanced Image", use_container_width=True)
    with col3:
        st.image(final_output, caption="Final Optimized Output", use_container_width=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image(mask_display, caption="Unsafe Region Mask", use_container_width=True)
    with col5:
        st.image(overlay, caption="Unsafe Regions Highlighted", use_container_width=True)
    with col6:
        st.image(heatmap, caption="Risk Heatmap", use_container_width=True)

    st.subheader("Final Analysis")
    st.success(f"CNN Support Classification: {pred_label}")
    st.info(f"CNN Confidence: {confidence:.2f}%")
    st.warning(f"Safety Score: {safety_score:.2f}/100")
    st.error(f"Risk Level: {risk_level}")

    st.markdown("### Recommendation")
    st.write(recommendation)