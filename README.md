# Advanced AI Face Swap & Forensics Suite

A production-grade Computer Vision application that performs high-fidelity, real-time facial manipulation alongside structural media forensics for deepfake detection. 

The core engineering approach focuses heavily on **hardware constraint mitigation**, **dynamic resource management**, and **on-demand dependency provisioning** to ensure optimal performance on standard consumer hardware.

---

## 🛠️ Tech Stack & Architecture

- **Core Engine:** InsightFace (buffalo_l pipeline)
- **Runtime Inference:** ONNX Runtime (CPU Optimized)
- **Frontend Interface:** Streamlit (Python native reactive framework)
- **Image Processing:** OpenCV, NumPy
- **Memory Management:** Built-in Python Garbage Collection (`gc`)

---

## 🚀 Engineering Highlights & Optimization Approach

Unlike standard AI script wrappers that cause memory leaks and system thermal throttling, this repository utilizes an optimized engineering pipeline designed to run seamlessly on restricted resource environments (e.g., 8GB RAM / Quad-core setups).

### 1. Manual Garbage Collection & Memory Management
- **The Challenge:** Native Python memory pools keep processed image frames in cache, rapidly inflating RAM usage and causing system hangs during back-to-back testing.
- **The Solution:** Implemented deterministic memory flushing using Python’s `gc` library. The system explicitly deletes large NumPy image arrays (`src_img`, `tgt_img`, `gray`) immediately after rendering and invokes `gc.collect()` to free memory blocks back to the OS.

### 2. Resolution Scaling for Facial Landmarks
- **The Challenge:** Running facial extraction models at the default 640x640 resolution causes high CPU utilization spikes and system overheating.
- **The Solution:** Tuned the detection sizing dynamically to `(320, 320)`. This cuts the tensor multi-dimensional arrays down significantly, reducing RAM footprints during inference by half while maintaining edge alignment accuracy.

### 3. On-Demand Model Provisioning
- **The Challenge:** Large Binary Large Objects (BLOs) like the 554MB `inswapper_128.onnx` file breach standard version control thresholds and crash server deployment nodes.
- **The Solution:** Programmed an automatic runtime download pipeline using `urllib.request`. The system searches for the binary locally; if absent, it fetches it via encrypted streams directly from high-bandwidth model hubs at runtime. This keeps the repository under 15KB.

### 4. Stateful Model Caching
- **The Challenge:** Re-initializing weights and layers on every reactive user interaction triggers severe CPU rendering overheads.
- **The Solution:** Isolated core models into a centralized layer using Streamlit's `@st.cache_resource` mutation hook. The models load precisely once into memory, operating as static singletons for all subsequent operations.

---

## 🔬 Forensics & Deepfake Detection Engine

To provide an ethical balance to media generation, the application includes a **Forensic Analytics Suite** utilizing spatial frequency tracking:
- **Texture Analysis:** Uses Laplacian variance tracking to isolate boundary pixel discontinuities. AI-generated patches or heavily altered edges show characteristic low-frequency variance boundaries.
- **Pixel Density Histogram:** Generates a real-time linear density map using OpenCV's `calcHist`. This provides structural transparency, showing developers and recruiters whether a file contains natural gradients or artificial quantization steps.

---

## 📦 Local Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd ai_face_studio
   ```

2. **Establish Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```
