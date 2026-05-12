import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps
import time

# --- 1. ASSET LOADING ---
# Ensure these files are in the same folder as app.py
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Grayscale -> Resize 120x100 -> Normalize -> Equalize (histeq)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    return img_eq.flatten().reshape(1, -1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR-X Multi-Target", layout="wide")
st.title("📡 GPR-X Intelligent Multi-Target Detector")
st.write("Upload a radargram to identify **Cavities**, **Bricks**, and **Metal Pipes**.")

# THE UPLOAD BUTTON - Placed here to ensure it's always visible
uploaded_file = st.file_uploader("Choose a GPR image (PNG, JPG, JPEG)...", type=["png", "jpg", "jpeg"])

# Sidebar for controls
st.sidebar.header("Scan Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.85)
stride_val = st.sidebar.select_slider("Scan Detail (Stride)", options=[20, 40, 60, 80], value=40)

if uploaded_file is not None:
    # Display the uploaded image immediately
    input_img = Image.open(uploaded_file).convert('RGB')
    st.image(input_img, caption="Uploaded Radargram", use_container_width=True)
    
    if st.button("🔍 Start Intelligent Scan"):
        if model is None:
            st.error("Cannot scan: Model files (svm_model.pkl) are missing!")
        else:
            # Running Detection
            w, h = input_img.size
            roi_w, roi_h = 120, 100
            detections = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Scanning loop
            y_steps = range(0, h - roi_h, stride_val)
            x_steps = range(0, w - roi_w, stride_val)
            total = len(y_steps) * len(x_steps)
            count = 0

            start_time = time.time()
            for y in y_steps:
                for x in x_steps:
                    count += 1
                    progress_bar.progress(count / total)
                    status_text.text(f"Scanning row {y}...")
                    
                    roi = input_img.crop((x, y, x + roi_w, y + roi_h))
                    feat = preprocess_roi(roi)
                    feat_scaled = scaler.transform(feat)
                    
                    probs = model.predict_proba(feat_scaled)[0]
                    cls_idx = np.argmax(probs)
                    conf = probs[cls_idx]
                    
                    if conf >= conf_threshold:
                        detections.append({'box': [x, y], 'class': cls_idx + 1, 'conf': conf})

            # Show Results
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(input_img)
            
            # Map labels to colors from your sources
            styles = {1: ("Cavity", "blue"), 2: ("Brick", "white"), 3: ("Metal", "cyan")}
            
            for d in detections:
                label, color = styles[d['class']]
                bx, by = d['box']
                rect = patches.Rectangle((bx, by), roi_w, roi_h, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(bx, by-10, f"{label} {d['conf']:.2f}", color=color, fontweight='bold', 
                        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
            
            plt.axis('off')
            st.pyplot(fig)
            st.success(f"Scan complete in {time.time()-start_time:.1f}s. Found {len(detections)} targets.")
else:
    st.info("Waiting for image upload...")
