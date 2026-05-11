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
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading AI assets: {e}")
        return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

# --- 2. AUTOMATIC SCANNING ENGINE ---
def run_auto_detection(full_img, model, scaler):
    h, w = full_img.shape
    roi_h, roi_w = 100, 120
    # Stride of 40 is fast; change to 20 for more detailed (but slower) scanning
    stride = 40 
    
    detections = []
    
    progress_bar = st.progress(0)
    total_steps = ((h - roi_h) // stride) * ((w - roi_w) // stride)
    step = 0

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            # Extract window and apply BEMD-style detrending
            window = full_img[y:y+roi_h, x:x+roi_w]
            clean_window = detrend(detrend(window, axis=0), axis=1)
            
            # Flatten and force to 11,999 features to match SVM.ipynb
            features = clean_window.flatten()
            if len(features) >= 11999:
                features = features[:11999].reshape(1, -1)
                
                try:
                    features_scaled = scaler.transform(features)
                    pred = model.predict(features_scaled)[0]
                    
                    # Store if it matches one of your 3 classes
                    if pred in [1, 2, 3]:
                        detections.append({'x': x, 'y': y, 'class': pred})
                except:
                    continue
            
            step += 1
            if step % 5 == 0 and step <= total_steps:
                progress_bar.progress(step / total_steps)
                
    progress_bar.empty()
    return detections

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR-X Auto-Detector", layout="wide")
st.title("📡 GPR-X Automatic Target Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # Load and Pre-process Image
    raw_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(raw_img).astype(np.float64)
    # Normalize pixel values to 0-1 range to help the SVM
    display_img = mat2gray_python(img_array)

    if st.sidebar.button("🚀 Start Automatic Scan"):
        with st.spinner("Analyzing patterns in the radargram..."):
            found = run_auto_detection(display_img, model, scaler)
        
        # --- 4. DISPLAY RESULTS (CROPPED VIEW) ---
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        class_info = {
            1: ("Cavity", "#238636"), # Green
            2: ("Brick", "#d29922"),  # Yellow
            3: ("Metal", "#da3633")   # Red
        }

        if not found:
            st.warning("No targets identified. Try adjusting the image contrast or retraining with image samples.")
        else:
            for d in found:
                name, color = class_info[d['class']]
                rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
                ax.add_patch(rect)
                ax.text(d['x'], d['y']-5, name, color=color, fontweight='bold', fontsize=10)

        # Remove axes and white space for a clean "Cropped" look
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        
        if found:
            st.success(f"Detection complete! Found {len(found)} potential targets.")
elif not model:
    st.error("⚠️ AI Assets (svm_model.pkl / scaler.pkl) not found. Check your file names and GitHub directory.")
