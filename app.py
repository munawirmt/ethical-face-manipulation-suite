import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import requests
import gc

st.set_page_config(page_title="Advanced AI Face Studio", layout="centered")
st.title("Advanced AI Face Suite")

st.sidebar.header("AI Model Configuration")
det_thresh = st.sidebar.slider("Face Detection Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)

@st.cache_resource
def load_system_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ക്ലൗഡ് സർവറിൽ മോഡലുകൾ ഓട്ടോമാറ്റിക്കായി ഡൗൺലോഡ് ചെയ്യും
    app = FaceAnalysis(name='buffalo_l')
    
    # ഇൻസ്വാപ്പർ ഫയൽ വെരിഫിക്കേഷൻ
    model_path = os.path.join(current_dir, 'inswapper_128.onnx')
    if os.path.exists(model_path) and os.path.getsize(model_path) < 500 * 1024 * 1024:
        os.remove(model_path)
        
    if not os.path.exists(model_path):
        with st.spinner("Downloading main execution engines... Please wait 1 minute."):
            url = 'https://huggingface.co'
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk: f.write(chunk)
                        
    swapper = insightface.model_zoo.get_model(model_path, download=False)
    return app, swapper

app, swapper = load_system_models()
app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=det_thresh)

tab1, tab2 = st.tabs(["Face Swap", "Forensics Detection"])

with tab1:
    st.subheader("Safe Image Face-Swap Pipeline")
    src_file = st.file_uploader("Source Face", type=["jpg", "png", "jpeg"], key="src_face")
    tgt_file = st.file_uploader("Target Face", type=["jpg", "png", "jpeg"], key="tgt_face")
    
    if src_file and tgt_file:
        if st.button("Run Face Swap"):
            with st.spinner("Processing safely..."):
                src_bytes = np.asarray(bytearray(src_file.read()), dtype=np.uint8)
                src_img = cv2.imdecode(src_bytes, cv2.IMREAD_COLOR)
                tgt_bytes = np.asarray(bytearray(tgt_file.read()), dtype=np.uint8)
                tgt_img = cv2.imdecode(tgt_bytes, cv2.IMREAD_COLOR)
                
                src_faces = app.get(src_img)
                tgt_faces = app.get(tgt_img)
                
                if len(src_faces) > 0 and len(tgt_faces) > 0:
                    output = tgt_img.copy()
                    output = swapper.get(output, tgt_faces, src_faces, paste_back=True)
                    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Result", width=400)
                    
                    is_success, buffer = cv2.imencode('.jpg', output)
                    if is_success:
                        st.download_button(label="📥 Download Result", data=buffer.tobytes(), file_name="output.jpg", mime="image/jpeg")
                else:
                    st.error("Face not detected. Adjust Confidence settings.")
                del src_img, tgt_img
                gc.collect()

with tab2:
    st.subheader("Deepfake Detection & Media Forensics")
    analysis_file = st.file_uploader("Upload Image to Scan", type=["jpg", "png", "jpeg"], key="detect_face")
    if analysis_file:
        if st.button("Scan Media"):
            with st.spinner("Analyzing..."):
                file_bytes = np.asarray(bytearray(analysis_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                st.metric(label="Texture Consistency Score", value=f"{texture_score:.2f}")
                
                if texture_score < 100: st.error("Result: HIGH RISK")
                else: st.success("Result: LOW RISK")
                
                st.subheader("Pixel Density Analysis")
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                st.line_chart(hist)
                del img, gray, hist
                gc.collect()
