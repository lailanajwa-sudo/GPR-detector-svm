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
        st.error(f"Error loading assets: {e}")
        return None, None

model, scaler = load_assets()

def mat2gray_python(img):
    mn, mx = np.min(img), np.max(img)
    diff = mx - mn
    return (img - mn) / diff if diff > 1e-7 else np.zeros_like(img)

# --- 2. PRECISE AUTO-DETECTION ENGINE ---
def run_precise_scan(img_array, model, scaler, threshold=0.80):
    h, w = img_array.shape
    roi_h, roi_w = 100, 120
    stride = 40 # Adjust for speed/precision. Lower is more precise.
    
    detections = []
    
    # Standardize image to match the "signal" distribution learned by the SVM
    # This helps the model recognize patterns in pixel data.
    img_std = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    for y in range(0, h - roi_h, stride):
        for x in range(0, w - roi_w, stride):
            # Extract and filter ROI
            window = img_std[y:y+roi_h, x:x+roi_w]
            clean_window = detrend(detrend(window, axis=0), axis=1)
            
            # FLATTEN TO EXACTLY 12,000 FEATURES
            # This fixes the ValueError: StandardScaler is expecting 12000 features.
            features = clean_window.flatten().reshape(1, -1)
            
            if features.shape[1] == 12000:
                # Use probabilities to filter out "weak" detections
                probs = model.predict_proba(scaler.transform(features))[0]
                best_class_idx = np.argmax(probs)
                confidence = probs[best_class_idx]
                
                # Classes: 1=Cavity, 2=Brick, 3=Metal
                if confidence >= threshold:
                    detections.append({
                        'x': x, 'y': y, 
                        'class': best_class_idx + 1, 
                        'conf': confidence
                    })
    return detections

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR-X Precise Detector", layout="wide")
st.title("📡 GPR-X Precise Automatic Detection")

uploaded_file = st.sidebar.file_uploader("Upload Radargram Image", type=["jpg", "jpeg", "png"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.50, 0.99, 0.85)

if uploaded_file and model:
    # Convert image to grayscale for processing
    raw_img = Image.open(uploaded_file).convert('L')
    img_np = np.array(raw_img).astype(np.float64)
    display_img = mat2gray_python(img_np)

    if st.sidebar.button("🔍 Run Intelligent Scan"):
        with st.spinner("Analyzing radargram for targets..."):
            found = run_precise_scan(img_np, model, scaler, conf_threshold)
        
        # Display Results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(display_img, cmap='gray', aspect='auto')
        
        # Color mapping to match your requested style
        class_info = {
            1: ("Cavity", "#0000FF"), # Blue
            2: ("Brick", "#FFFFFF"),  # White
            3: ("Metal", "#00FFFF")   # Cyan
        }

        if not found:
            st.warning("No targets detected above the confidence threshold.")
        else:
            for d in found:
                name, color = class_info[d['class']]
                # Draw bounding box and label with confidence
                rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=color, fill=False)
                ax.add_patch(rect)
                ax.text(d['x'], d['y']-5, f"{name} {d['conf']:.2f}", color=color, fontweight='bold', fontsize=10)

        # Final "Crop" to focus only on the radargram
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        st.pyplot(fig, use_container_width=True)
        
        if found:
            st.success(f"Scan complete. Found {len(found)} targets.")

elif not model:
    st.error("⚠️ AI Assets not found. Please ensure svm_model.pkl and scaler.pkl are in the same directory.")
