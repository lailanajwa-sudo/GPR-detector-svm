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
        # Loading the SVM model and scaler trained in Colab
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    # Grayscale -> Resize to 120x100 to ensure exactly 12,000 features
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray: Scale pixels between 0 and 1
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Histogram Equalization: Matches the MATLAB histeq logic to boost signal edges
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    
    return img_eq.flatten().reshape(1, -1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR Signal Analyzer", layout="wide")
st.title("📡 GPR Hyperbola Signal Analyzer")

uploaded_file = st.file_uploader("Upload Radargram (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    st.subheader("Targeting Controls")
    st.write("Move the sliders to frame a hyperbola, then click Analyze.")
    
    col_x, col_y = st.columns(2)
    with col_x:
        # Use a large step or just default to make it smoother
        pos_x = st.slider("X Position", 0, w - 120, int(w/2))
    with col_y:
        pos_y = st.slider("Y Position", 0, h - 100, int(h/2))
    
    # PREVIEW (Updates instantly because no BEMD is running yet)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    # Yellow dashed box shows the user where the AI will look
    rect = patches.Rectangle((pos_x, pos_y), 120, 100, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # ANALYSIS TRIGGER
    if st.button("🚀 Analyze Signal in Box"):
        roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
        
        # Calculate standard deviation to detect 'Background' vs 'Signal'
        roi_std = np.std(np.array(roi))
        
        if roi_std < 8:
            st.warning("⚠️ Area is too flat. No clear hyperbolic signal detected in this box.")
        else:
            with st.spinner('Extracting BEMD features...'):
                # Processing features for the SVM
                features = preprocess_roi(roi)
                features_scaled = scaler.transform(features)
                
                # Get probabilities for each class
                probs = model.predict_proba(features_scaled)[0]
                classes = ["Cavity", "Brick", "Metal Pipe"]
                best_idx = np.argmax(probs)
                
                st.divider()
                res_col, bar_col = st.columns(2)
                
                with res_col:
                    st.success(f"**Classification:** {classes[best_idx]}")
                    st.metric("Confidence Score", f"{probs[best_idx]*100:.1f}%")
                    
                with bar_col:
                    for i, p in enumerate(probs):
                        st.write(f"{classes[i]}: {p*100:.1f}%")
                        st.progress(float(p))
