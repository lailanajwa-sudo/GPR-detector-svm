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
    # This is the secret: No matter how big/small your box is, 
    # we resize it to 120x100 so it matches the 12,000 training features.
    roi_gray = ImageOps.grayscale(roi_img).resize((120, 100))
    img_np = np.array(roi_gray).astype(np.float64)
    
    # mat2gray normalization
    img_norm = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-7)
    
    # Flatten and Scale using the StandardScaler from training
    flat_data = img_norm.flatten().reshape(1, -1)
    return scaler.transform(flat_data)

# --- 3. UI LAYOUT ---
st.title("📡 Precision GPR Analyzer")

uploaded_file = st.file_uploader("Upload Radargram", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert('RGB')
    w, h = img.size
    
    # CONTROLS
    st.subheader("Targeting Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pos_x = st.number_input("X Pos", 0, w, int(w/2))
    with c2:
        pos_y = st.number_input("Y Pos", 0, h, int(h/2))
    with c3:
        # NEW: Adjust box width
        box_w = st.slider("Box Width", 20, 300, 80)
    with c4:
        # NEW: Adjust box height
        box_h = st.slider("Box Height", 20, 300, 60)
    
    # PREVIEW
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img)
    # The box now reflects your slider choices
    rect = patches.Rectangle((pos_x, pos_y), box_w, box_h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.axis('off')
    st.pyplot(fig)

    # ANALYZE
    if st.button("🚀 Analyze Target"):
        # Crop using the custom dimensions
        roi = img.crop((pos_x, pos_y, pos_x + box_w, pos_y + box_h))
        
        # Standard Deviation Check (Filter out background)
        if np.std(np.array(roi)) < 10:
            st.warning("⚠️ Area is too blank. Move the box over a hyperbola.")
        else:
            features = preprocess_roi(roi)
            probs = model.predict_proba(features)[0]
            classes = ["Cavity", "Brick", "Metal Pipe"]
            res = np.argmax(probs)
            
            st.divider()
            st.success(f"**Result:** {classes[res]} ({probs[res]*100:.1f}%)")
            
            # Show specific breakdown
            for i, p in enumerate(probs):
                st.write(f"{classes[i]}: {p*100:.1f}%")
                st.progress(float(p))
