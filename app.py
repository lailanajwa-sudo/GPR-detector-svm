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
        st.error("Model files (svm_model.pkl/scaler.pkl) not found!")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Fixed resize to match your 12,000 feature training (120x100)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    # Normalize 0-1 (mat2gray)
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    # Flatten and transform with the training scaler
    features = img_norm.flatten().reshape(1, -1)
    return scaler.transform(features)

# --- 3. UI SETUP ---
st.set_page_config(layout="centered") 
st.title("📡 GPR Target Classifier")

uploaded_file = st.file_uploader("Step 1: Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # FIXED BOX DIMENSIONS (Adjust these numbers if the red box is still too big)
    BW, BH = 80, 60 

    # POSITION SLIDERS
    st.write("### Step 2: Position the Box over a Hyperbola")
    pos_x = st.slider("Horizontal Position (X)", 0, w - BW, int(w/2))
    pos_y = st.slider("Vertical Position (Y)", 0, h - BH, int(h/2))

    # SMALLER CENTERED PREVIEW
    _, mid_col, _ = st.columns([1, 4, 1]) 
    with mid_col:
        fig, ax = plt.subplots(figsize=(5, 3)) # Small figure size
        ax.imshow(img)
        # Red bounding box
        rect = patches.Rectangle((pos_x, pos_y), BW, BH, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(fig)

    # ANALYZE BUTTON
    if st.button("🚀 ANALYZE SELECTED AREA", use_container_width=True):
        roi = img.crop((pos_x, pos_y, pos_x + BW, pos_y + BH))
        
        # Background Noise Filter
        if np.std(np.array(roi)) < 8:
            st.warning("No signal detected. Please move the box to a hyperbola peak.")
        else:
            with st.spinner('Analyzing...'):
                features_scaled = preprocess_roi(roi)
                probs = model.predict_proba(features_scaled)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                idx = np.argmax(probs)
                
                st.divider()
                st.success(f"### Result: {classes[idx]}")
                st.write(f"**Confidence:** {probs[idx]*100:.1f}%")
                
                # Probability breakdown
                for i, p in enumerate(probs):
                    st.write(f"{classes[i]}: {p*100:.1f}%")
                    st.progress(float(p))
