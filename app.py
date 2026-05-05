# 1. Parse .rad for SAMPLES count (312 in your case) 
rad_content = rad_file.getvalue().decode("utf-8")
samples_val = 312 
for line in rad_content.split('\n'):
    if "SAMPLES:" in line:
        samples_val = int(line.split(':')[1].strip())

# 2. Read .rd3 Binary Data
raw_data = np.frombuffer(rd3_file.read(), dtype=np.int16).astype(float)
num_traces = len(raw_data) // samples_val

if num_traces > 0:
    # 3. Reshape with 'F' order
    matrix = raw_data[:samples_val*num_traces].reshape((samples_val, num_traces), order='F')
    
    # --- THE FIX FOR "TERBALIK" AND VISUAL STYLE ---
    
    # REMOVE np.flipud() -> Keeping it original puts the surface at the top
    matrix_correct_orient = matrix 
    
    # Apply Background Removal (Dewow/Subtract Mean)
    matrix_clean = matrix_correct_orient - np.mean(matrix_correct_orient, axis=1, keepdims=True)
    
    # Replicate MATLAB mat2gray/contrast
    # We use 99th percentile to make it clean like Cav001.png
    limit = np.percentile(np.abs(matrix_clean), 99)

    # 4. Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use 'gray' colormap to match your target
    img = ax.imshow(matrix_clean, 
                    cmap='gray', 
                    aspect='auto', 
                    vmin=-limit, 
                    vmax=limit)
    
    # Set labels to match Cav001.png
    ax.set_ylabel("Time (samples)")
    ax.set_xlabel("Trace")
    ax.set_title("Radargram Reconstructed")
    
    st.pyplot(fig)
