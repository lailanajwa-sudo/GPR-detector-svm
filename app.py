import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from PIL import Image
import time

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    try:
        model = joblib.load(os.path.join(base_path, 'svm_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

def mat2gray(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn + 1e-7)

# --- 2. INTELLIGENT SCAN ENGINE ---
def run_intelligent_scan(img_array, model, scaler, threshold=0.90):
    h, w = img_array.shape
    roi_h, roi_w = 100, 120
    # Stride 60 is fast and effective for hyperbolas
    stride = 60 
    
    detections = []
    
    # NORMALIZATION: This helps the AI distinguish Cavity from Metal
    # by centering the pixel intensities around zero.
    img_std = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    y_steps = list(range(0, h - roi_h, stride))
    
    # Progress Indicators
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i, y in enumerate(y_steps):
        # Update the loading line
        percent = (i + 1) / len(y_steps)
        progress_bar.progress(percent)
        status_text.text(f"Scanning Radargram: {int(percent*100)}% complete...")

        for x in range(0, w - roi_w, stride):
            window = img_std[y:y+roi_h, x:x+roi_w]
            # BEMD Filter equivalent (Detrending)
            clean = detrend(detrend(window, axis=0), axis=1)
            
            # Use exactly 12,000 features to match StandardScaler
            features = clean.flatten().reshape(1, -1)
            
            if features.shape[1] == 12000:
                # Use Probability to filter False Positives
                probs = model.predict_proba(scaler.transform(features))[0]
                best_class_idx = np.argmax(probs)
                confidence = probs[best_class_idx]
                
                # Check confidence threshold
                if confidence >= threshold:
                    detections.append({
                        'x': x, 'y': y, 
                        'class': best_class_idx + 1, 
                        'conf': confidence
                    })
    
    progress_bar.empty()
    status_text.empty()
    return detections

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR-X Precision", layout="wide")
st.title("📡 GPR-X Precise Auto-Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "png", "jpeg"])
# Slider to control how strict the AI is
conf_limit = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.90)

if uploaded_file and model:
    # Prepare image
    img = Image.open(uploaded_file).convert('L')
    img_np = np.array(img).astype(np.float64)
    display_img = mat2gray(img_np)

    if st.sidebar.button("🚀 Start Precision Scan"):
        start_time = time.time()
        found = run_intelligent_scan(img_np, model, scaler, conf_limit)
        duration = time.time() - start_time
        
        # Display Results
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Style mapping to match your requested output
        styles = {
            1: ("cavity", "#0000FF"),      # Blue
            2: ("brick", "#FFFFFF"),       # White
            3: ("metal_pipe", "#00FFFF")   # Cyan
        }

        if not found:
            st.warning("No targets detected. Try lowering the Confidence Threshold.")
        else:
            for d in found:
                name, color = styles[d['class']]
                # Draw professional bounding box
                rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
                ax.add_patch(rect)
                # Add label with confidence score
                ax.text(d['x'], d['y']-5, f"{name} {d['conf']:.2f}", color=color, weight='bold', fontsize=10, 
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        # True Crop: Remove axes and white margins
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        
        st.success(f"Scan finished in {duration:.1f}s. {len(found)} objects identified.")

elif not model:
    st.error("AI Assets (svm_model.pkl / scaler.pkl) not found in directory!")
