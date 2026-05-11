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
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    try:
        # Loading the model and scaler trained in SVM.ipynb
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

def mat2gray(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn + 1e-7)

# --- 2. INTELLIGENT SCAN ENGINE WITH PROGRESS BAR ---
def run_precise_scan(img_array, model, scaler, threshold=0.85):
    h, w = img_array.shape
    roi_h, roi_w = 100, 120
    stride = 50 # Balanced speed and accuracy
    
    detections = []
    img_std = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    # Setup for the loading line
    y_steps = list(range(0, h - roi_h, stride))
    total_steps = len(y_steps)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, y in enumerate(y_steps):
        # Update loading line status
        percent_complete = (i + 1) / total_steps
        progress_bar.progress(percent_complete)
        progress_text.text(f"Scanning Depth: {int(percent_complete * 100)}% complete...")

        for x in range(0, w - roi_w, stride):
            window = img_std[y:y+roi_h, x:x+roi_w]
            clean = detrend(detrend(window, axis=0), axis=1)
            
            # Use exactly 12000 features to match Scaler
            features = clean.flatten().reshape(1, -1)
            
            if features.shape[1] == 12000:
                # Get probabilities for clean boxes
                probs = model.predict_proba(scaler.transform(features))[0]
                best_class = np.argmax(probs)
                confidence = probs[best_class]
                
                if confidence >= threshold:
                    detections.append({
                        'x': x, 'y': y, 
                        'class': best_class + 1, 
                        'conf': confidence
                    })
    
    # Clear the loading line once finished
    progress_bar.empty()
    progress_text.empty()
    return detections

# --- 3. UI ---
st.set_page_config(page_title="GPR-X Precision", layout="wide")
st.title("📡 GPR-X Precise Auto-Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram (JPG/PNG)", type=["jpg", "png", "jpeg"])
conf_val = st.sidebar.slider("Confidence (Threshold)", 0.50, 0.99, 0.85)

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('L')
    img_np = np.array(img).astype(np.float64)
    display_img = mat2gray(img_np)

    if st.sidebar.button("🚀 Start Precision Scan"):
        start_time = time.time()
        found = run_precise_scan(img_np, model, scaler, conf_val)
        duration = time.time() - start_time
        
        # Display Results
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Styles to match your hyperbola detection request
        styles = {
            1: ("cavity", "#0000FF"),      # Blue
            2: ("brick", "#FFFFFF"),       # White
            3: ("metal_pipe", "#00FFFF")   # Cyan
        }

        for d in found:
            name, color = styles[d['class']]
            # Draw professional bounding box
            rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
            ax.add_patch(rect)
            # Label with confidence like your 2nd image
            ax.text(d['x'], d['y']-5, f"{name} {d['conf']:.2f}", color=color, weight='bold', fontsize=10, 
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        st.success(f"Scan complete in {duration:.1f}s. Found {len(found)} objects.")

elif not model:
    st.error("AI Assets (svm_model.pkl / scaler.pkl) not found!")
