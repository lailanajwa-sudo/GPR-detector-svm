import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Model files not found!")
        return None, None

model, scaler = load_assets()

# --- 2. UI SETUP ---
st.set_page_config(layout="centered") # Keeps the whole app narrow
st.title("📡 GPR Pattern Classifier")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # 1. ANALYZE BUTTON AT TOP
    analyze_btn = st.button("🚀 ANALYZE SIGNAL", use_container_width=True)

    # 2. CONTROLS (Using columns to save space)
    st.write("### Targeting Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1: x = st.number_input("X", 0, w, int(w/2))
    with c2: y = st.number_input("Y", 0, h, int(h/2))
    with c3: bw = st.number_input("Width", 10, w, 80)
    with c4: bh = st.number_input("Height", 10, h, 60)

    # 3. SMALLER PREVIEW
    # We create a specific column layout to center a small image
    _, mid_col, _ = st.columns([1, 3, 1]) 
    
    with mid_col:
        # We set a smaller figsize (e.g., 5x3 inches instead of 10x5)
        fig, ax = plt.subplots(figsize=(6, 4)) 
        ax.imshow(img)
        rect = patches.Rectangle((x, y), bw, bh, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        # use_container_width=True here will fill the 3/5 width of the screen
        st.pyplot(fig, use_container_width=True)

    # 4. LOGIC
    if analyze_btn:
        x_end, y_end = min(x + bw, w), min(y + bh, h)
        roi = img.crop((x, y, x_end, y_end))
        
        if np.std(np.array(roi)) < 5:
            st.warning("Selected area is too blank.")
        else:
            # Resize for SVM (120x100 = 12,000 features)
            roi_gray = ImageOps.grayscale(roi).resize((120, 100))
            img_norm = (np.array(roi_gray) - np.min(roi_gray)) / (np.max(roi_gray) - np.min(roi_gray) + 1e-7)
            
            features = scaler.transform(img_norm.flatten().reshape(1, -1))
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            
            st.divider()
            st.success(f"### Detected: {classes[np.argmax(probs)]}")
            st.write(f"Confidence: {np.max(probs)*100:.1f}%")
