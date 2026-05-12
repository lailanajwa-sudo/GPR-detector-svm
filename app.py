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
        st.error("Model files not found! Please upload svm_model.pkl and scaler.pkl to your GitHub.")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Resize to exactly 120x100 to match your 12,000 features
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    # Normalize 0-1
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    flat = img_norm.flatten().reshape(1, -1)
    return scaler.transform(flat)

# --- 3. UI ---
st.title("📡 GPR Pattern Classifier")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # Place button at the top so it's always visible
    analyze_btn = st.button("🚀 CLICK HERE TO ANALYZE", use_container_width=True)

    st.divider()
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x = st.number_input("X Pos", 0, w, int(w/2))
    with col2:
        y = st.number_input("Y Pos", 0, h, int(h/2))
    with col3:
        bw = st.number_input("Box Width", 10, w, 80)
    with col4:
        bh = st.number_input("Box Height", 10, h, 60)

    # PREVIEW
    fig, ax = plt.subplots()
    ax.imshow(img)
    # Red box for better visibility
    rect = patches.Rectangle((x, y), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # LOGIC RUNS WHEN BUTTON IS CLICKED
    if analyze_btn:
        # Safety crop to stay inside image boundaries
        x_end = min(x + bw, w)
        y_end = min(y + bh, h)
        roi = img.crop((x, y, x_end, y_end))
        
        # Check if the area is valid (not just a single color)
        if np.std(np.array(roi)) < 5:
            st.warning("The selected area is too blank (background). Move the box to a hyperbola.")
        else:
            with st.spinner('Classifying...'):
                features = preprocess_roi(roi)
                probs = model.predict_proba(features)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                idx = np.argmax(probs)
                
                st.balloons()
                st.success(f"### RESULT: {classes[idx]}")
                st.write(f"Confidence: {probs[idx]*100:.1f}%")
                
                # Show all probabilities
                for i, p in enumerate(probs):
                    st.write(f"{classes[i]}: {p*100:.2f}%")
                    st.progress(float(p))
