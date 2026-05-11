import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import detrend
from PIL import Image

# --- 1. LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'svm_model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            st.error("Fail .pkl tidak dijumpai! Pastikan svm_model.pkl & scaler.pkl ada dalam GitHub.")
            return None, None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

model, scaler = load_assets()

def mat2gray(img):
    mn, mx = np.min(img), np.max(img)
    return (img - mn) / (mx - mn + 1e-7)

# --- 2. ENGINE AUTO-DETECTION ---
def scan_radargram(img_array, model, scaler):
    h, w = img_array.shape
    roi_h, roi_w = 100, 120
    stride = 30  # Jarak lompatan scanner (kecil = teliti, besar = laju)
    
    results = []
    prog = st.progress(0)
    
    # Tukar imej ke skala yang mirip dengan data training (Scaling manual)
    # Ini penting supaya SVM kenal pixel sebagai signal
    img_standardized = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)

    y_range = range(0, h - roi_h, stride)
    total = len(y_range)
    
    for i, y in enumerate(y_range):
        for x in range(0, w - roi_w, stride):
            # Ambil kotak ROI
            window = img_standardized[y:y+roi_h, x:x+roi_w]
            
            # Feature Extraction (BEMD / Detrend)
            clean = detrend(detrend(window, axis=0), axis=1)
            
            # Flatten ke 11,999 features (Ikut SVM.ipynb anda)
            feat = clean.flatten()[:11999].reshape(1, -1)
            
            # Predict
            feat_scaled = scaler.transform(feat)
            pred = model.predict(feat_scaled)[0]
            
            if pred in [1, 2, 3]: # Jika jumpa Cavity, Brick, atau Metal
                results.append({'x': x, 'y': y, 'class': pred})
        
        prog.progress((i + 1) / total)
    
    prog.empty()
    return results

# --- 3. UI STREAMLIT ---
st.set_page_config(page_title="GPR-X Auto Detector", layout="wide")
st.title("📡 GPR-X Auto-Detection (JPG/PNG Mode)")

file = st.sidebar.file_uploader("Upload Gambar Radargram", type=["jpg", "jpeg", "png"])

if file and model:
    # Proses Imej
    img = Image.open(file).convert('L') # Tukar ke Grayscale
    img_np = np.array(img).astype(np.float64)
    
    if st.sidebar.button("🔍 Mula Scan Automatik"):
        with st.spinner("Sedang mencari sasaran..."):
            detections = scan_radargram(img_np, model, scaler)
        
        # Display Result
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img_np, cmap='gray', aspect='auto')
        
        labels = {1: ("Cavity", "green"), 2: ("Brick", "yellow"), 3: ("Metal", "red")}
        
        for d in detections:
            txt, clr = labels[d['class']]
            rect = patches.Rectangle((d['x'], d['y']), 120, 100, linewidth=2, edgecolor=clr, fill=False)
            ax.add_patch(rect)
            ax.text(d['x'], d['y']-5, txt, color=clr, fontsize=10, weight='bold')

        # CROP VIEW: Buang margin putih & axis
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        st.pyplot(fig, use_container_width=True)
        
        if detections:
            st.success(f"Jumpa {len(detections)} sasaran!")
        else:
            st.warning("Tiada sasaran dijumpai. Cuba imej yang lebih jelas atau gerakkan kotak manual.")
