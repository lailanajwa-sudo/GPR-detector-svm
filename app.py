import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from PIL import Image

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading AI assets: {e}")
        return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

# --- 2. AUTOMATIC SCANNING LOGIC ---
def auto_detect(full_img, model, scaler):
    h, w = full_img.shape
    roi_h, roi_w = 100, 120
    stride = 40  # How many pixels to skip per step (lower = more accurate but slower)
    
    found_boxes = []
    
    # Progress bar for the scan
    progress_bar = st.progress(0)
    total_steps = ((h - roi_h) // stride) * ((w - roi_w) // stride)
    step_count = 0

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            # 1. Extract Window
            window = full_img[y:y+roi_h, x:x+roi_w]
            
            # 2. Preprocess (BEMD Filter / Detrend)
            clean_window = detrend(detrend(window, axis=0), axis=1)
            features = clean_window.flatten().reshape(1, -1)
            
            # 3. Handle Feature Mismatch
            if features.shape[1] == 12000:
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]
                
                # If it detects Class 1, 2, or 3 (Ignore background if you have a 4th class or use logic)
                if pred in [1, 2, 3]:
                    found_boxes.append({'x': x, 'y': y, 'class': pred})
            
            step_count += 1
            if step_count % 10 == 0:
                progress_bar.progress(min(step_count / total_steps, 1.0))
                
    progress_bar.empty()
    return found_boxes

# --- 3. UI LAYOUT ---
st.title("📡 GPR-X Auto-Detector")

uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Load and Pre-process
    raw_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(raw_img).astype(np.float64)
    display_img = mat2gray_python(img_array)

    if st.sidebar.button("🚀 Start Automatic Detection"):
        with st.spinner("Scanning radargram for targets..."):
            detections = auto_detect(img_array, model, scaler)
            
        # --- 4. DISPLAY RESULTS ---
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        results_map = {
            1: ("Cavity", "#238636"),
            2: ("Brick", "#d29922"),
            3: ("Metal", "#da3633")
        }

        if not detections:
            st.warning("No targets detected in this scan.")
        else:
            for d in detections:
                label, color = results_map[d['class']]
                rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=1.5, edgecolor=color, fill=False)
                ax.add_patch(rect)
                ax.text(d['x'], d['y']-5, label, color=color, fontsize=8, fontweight='bold')

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig)
        st.success(f"Detection complete! Found {len(detections)} possible targets.")
