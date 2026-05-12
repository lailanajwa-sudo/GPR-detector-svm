import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        # Loading model and scaler from training
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Grayscale -> Resize to 120x100 for 12,000 features
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    # mat2gray equivalent
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    # Contrast boost
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    return img_eq.flatten().reshape(1, -1)

# --- 3. UI ---
st.title("📡 GPR Hyperbola Signal Analyzer")
uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # Targeting Sliders
    col_x, col_y = st.columns(2)
    with col_x:
        pos_x = st.slider("X Position", 0, w - 120, int(w/2))
    with col_y:
        pos_y = st.slider("Y Position", 0, h - 100, int(h/2))
    
    # Preview Image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    rect = patches.Rectangle((pos_x, pos_y), 120, 100, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # Analysis Trigger
    if st.button("🚀 Analyze Signal"):
        roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
        
        # SIGNAL FILTER: Stops "Only Cavity" errors on blank backgrounds
        if np.std(np.array(roi)) < 10:
            st.warning("⚠️ No clear signal detected. This area appears to be background/soil.")
        else:
            features = preprocess_roi(roi)
            features_scaled = scaler.transform(features)
            
            probs = model.predict_proba(features_scaled)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            best_idx = np.argmax(probs)
            
            st.divider()
            if probs[best_idx] > 0.80:
                st.success(f"**Detected:** {classes[best_idx]} ({probs[best_idx]*100:.1f}%)")
            else:
                st.warning(f"**Weak Detection:** Likely {classes[best_idx]}. Try centering the hyperbola.")
                
            for i, p in enumerate(probs):
                st.write(f"{classes[i]}: {p*100:.1f}%")
                st.progress(float(p))
