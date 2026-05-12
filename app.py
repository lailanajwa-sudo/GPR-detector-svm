import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("Missing svm_model.pkl or scaler.pkl")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Grayscale and Resize to 120x100 (12,000 features)
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    # mat2gray normalization
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    # Flatten and Scale
    flat = img_norm.flatten().reshape(1, -1)
    return scaler.transform(flat)

# --- 3. UI ---
st.title("📡 GPR Signal Analyzer")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    st.info("Set the coordinates and click Analyze to see the classification.")
    
    col1, col2 = st.columns(2)
    with col1:
        pos_x = st.number_input("X Coordinate", 0, w - 120, int(w/2))
    with col2:
        pos_y = st.number_input("Y Coordinate", 0, h - 100, int(h/2))

    # ONLY SHOW THE BUTTON - NO LIVE PREVIEW
    if st.button("🚀 ANALYZE SIGNAL"):
        roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
        
        # Check for empty background (Variance check)
        if np.std(np.array(roi)) < 12:
            st.warning("⚠️ Area appears to be empty background. Try a different coordinate.")
        else:
            # Prediction
            features = preprocess_roi(roi)
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            idx = np.argmax(probs)
            
            # Show Result
            st.divider()
            st.success(f"**Classification:** {classes[idx]}")
            st.metric("Confidence", f"{probs[idx]*100:.1f}%")
            
            # Show the visual proof of where it looked
            fig, ax = plt.subplots()
            ax.imshow(img)
            rect = patches.Rectangle((pos_x, pos_y), 120, 100, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            plt.axis('off')
            st.pyplot(fig)
            
            # Show detail bars
            for i, p in enumerate(probs):
                st.write(f"{classes[i]}: {p*100:.1f}%")
                st.progress(float(p))
