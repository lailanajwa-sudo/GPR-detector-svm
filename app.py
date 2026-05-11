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
        # Load the model and scaler trained in your SVM.ipynb
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

def mat2gray(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn + 1e-7)

# --- 2. OPTIMIZED SCANNING ENGINE ---
def run_intelligent_scan(img_array, model, scaler, threshold=0.90):
    h, w = img_array.shape
    roi_h, roi_w = 100, 120
    # Increased stride to 60 for much faster performance
    stride = 60 
    
    detections = []
    # Standardize image to match signal distribution from SVM.ipynb
    img_std = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            window = img_std[y:y+roi_h, x:x+roi_w]
            # BEMD-style filtering
            clean = detrend(detrend(window, axis=0), axis=1)
            
            # Flatten to exactly 12,000 features to avoid ValueError
            features = clean.flatten().reshape(1, -1)
            
            if features.shape[1] == 12000:
                # Use probabilities for clean boxes like your screenshot
                probs = model.predict_proba(scaler.transform(features))[0]
                best_class = np.argmax(probs)
                confidence = probs[best_class]
                
                # Only keep detections that meet your confidence threshold
                if confidence >= threshold:
                    detections.append({
                        'x': x, 'y': y, 
                        'class': best_class + 1, 
                        'conf': confidence
                    })
    return detections

# --- 3. UI ---
st.set_page_config(page_title="GPR-X Precision", layout="wide")
st.title("📡 GPR-X Precision Auto-Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram", type=["jpg", "png", "jpeg"])
conf_val = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.90)

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('L')
    img_np = np.array(img).astype(np.float64)
    display_img = mat2gray(img_np)

    if st.sidebar.button("🔍 Run Precision Scan"):
        found = run_intelligent_scan(img_np, model, scaler, conf_val)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Color styles to match your request
        styles = {
            1: ("cavity", "#0000FF"), # Blue
            2: ("brick", "#FFFFFF"),  # White
            3: ("metal_pipe", "#00FFFF") # Cyan
        }

        for d in found:
            name, color = styles[d['class']]
            # Draw the professional bounding box
            rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
            ax.add_patch(rect)
            # Add label with confidence score (e.g., cavity 0.89)
            ax.text(d['x'], d['y']-5, f"{name} {d['conf']:.2f}", color=color, weight='bold', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.5, edgecolor='none', pad=1))

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        st.success(f"Scan complete: {len(found)} objects detected.")
