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
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- 2. PREPROCESSING ---
def preprocess_roi(roi_img):
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # Normalization
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Contrast Enhancement (histeq)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_eq = np.array(ImageOps.equalize(Image.fromarray(img_uint8))).astype(np.float64) / 255.0
    
    return img_eq.flatten().reshape(1, -1)

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="GPR Professional Classifier", layout="wide")
st.title("📡 Precision GPR Target Analyzer")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # 1. SLIDERS (Using 'key' to prevent laggy re-runs)
    st.subheader("Targeting Controls")
    col_x, col_y = st.columns(2)
    with col_x:
        pos_x = st.slider("X Position", 0, w - 120, int(w/2), key="x_slider")
    with col_y:
        pos_y = st.slider("Y Position", 0, h - 100, int(h/2), key="y_slider")
    
    # 2. THE IMAGE PREVIEW (Updates instantly with sliders)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    rect = patches.Rectangle((pos_x, pos_y), 120, 100, linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # 3. MANUAL TRIGGER (Fixes the "Slow Slider" issue)
    if st.button("🚀 Analyze Signal in Box"):
        roi = img.crop((pos_x, pos_y, pos_x + 120, pos_y + 100))
        
        # --- SIGNAL CHECK (Fixes the "Only Cavity" issue) ---
        # If the box is too 'flat' (standard deviation is low), it's just background.
        if np.std(np.array(roi)) < 8:
            st.warning("⚠️ No clear signal detected. The box appears to be empty soil/background.")
        else:
            features = preprocess_roi(roi)
            features_scaled = scaler.transform(features)
            
            probs = model.predict_proba(features_scaled)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            best_idx = np.argmax(probs)
            
            # Confidence Threshold: If below 70%, don't trust it.
            if probs[best_idx] < 0.70:
                st.error("❓ Signal unclear. Try aligning the hyperbola peak exactly in the center.")
            else:
                res_col, bar_col = st.columns(2)
                with res_col:
                    st.success(f"**Detection:** {classes[best_idx]}")
                    st.metric("Confidence", f"{probs[best_idx]*100:.1f}%")
                with bar_col:
                    for i, p in enumerate(probs):
                        st.write(f"{classes[i]}: {p*100:.1f}%")
                        st.progress(float(p))

### Why this fixes your problems:

1.  **Fast Sliders**: By moving the AI prediction inside an `if st.button` block, the app no longer tries to run BEMD while you are sliding. The slider will now feel smooth and instant.
2.  **`np.std` Filter**: The reason you keep getting "Cavity" is that the model is trying to classify **empty grey pixels**. Empty background looks most like a "Cavity" to the SVM. This filter checks if there is actually a "shape" (variance) in the box before guessing.
3.  **Centered Alignment**: For **Brick** and **Metal Pipe**, the SVM expects to see the "Peak" of the hyperbola. Use the sliders to place the **top curve** of the hyperbola right in the center of the yellow dashed box before clicking "Analyze."
