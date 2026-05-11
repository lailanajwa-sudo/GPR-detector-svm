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
    # Use absolute paths to ensure files are found in the GitHub/Streamlit environment
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error(f"❌ Files not found at: {base_path}. Please upload svm_model.pkl and scaler.pkl to your repository.")
            return None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
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
    # Stride determines speed vs accuracy. 40-60 is fast; 20 is very accurate but slow.
    stride = 40 
    
    detections = []
    
    # Simple progress bar for the scan
    progress_bar = st.progress(0)
    total_steps = ((h - roi_h) // stride) * ((w - roi_w) // stride)
    step = 0

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            # Extract window
            window = full_img[y:y+roi_h, x:x+roi_w]
            
            # Feature Extraction (BEMD Filter / Detrend)
            clean_window = detrend(detrend(window, axis=0), axis=1)
            
            # Match features to training (Exactly 11,999 or 12,000 depending on notebook)
            # Your notebook showed X = df.values. Flatten and adjust to match scaler.
            features = clean_window.flatten().reshape(1, -1)
            
            # Adjust to 11999 if that is what your Colab StandardScaler expects
            if features.shape[1] > 11999:
                features = features[:, :11999]

            try:
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)[0]
                
                # If a class is detected (1, 2, or 3)
                if pred in [1, 2, 3]:
                    detections.append({'x': x, 'y': y, 'class': pred})
            except:
                pass
            
            step += 1
            if step % 5 == 0:
                progress_bar.progress(min(step / total_steps, 1.0))
                
    progress_bar.empty()
    return detections

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR-X Auto-Detector", layout="wide")
st.title("📡 GPR-X Automatic Target Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # 1. Load Image
    raw_img = Image.open(uploaded_file).convert('L')
    img_array = np.array(raw_img).astype(np.float64)
    display_img = mat2gray_python(img_array)

    if st.sidebar.button("🚀 Run Auto Scan"):
        with st.spinner("Analyzing radargram patterns..."):
            found = run_auto_detection(img_array, model, scaler)
        
        # 2. Display with Cropped Focus (No axes or white margins)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Label mapping
        class_info = {
            1: ("Cavity", "#238636"), # Green
            2: ("Brick", "#d29922"),  # Yellow
            3: ("Metal", "#da3633")   # Red
        }

        if not found:
            st.warning("No targets identified in this radargram.")
        else:
            for d in found:
                name, color = class_info[d['class']]
                # Draw bounding box
                rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
                ax.add_patch(rect)
                ax.text(d['x'], d['y']-5, name, color=color, fontweight='bold', fontsize=10)

        # Remove axes and margins to "crop" the view to just the data
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        
        if found:
            st.success(f"Detection complete! Identified {len(found)} potential targets.")
