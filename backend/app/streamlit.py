import streamlit as st
import cv2
import requests
import numpy as np
import tempfile
import time
import os

# ---------------------------------------------------------
# CONFIGURATION & UI SETUP
# ---------------------------------------------------------
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Vehicle Analytics | Glass Blue Edition",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Darkish Glassy Blue" aesthetic
st.markdown("""
    <style>
    /* Main Background: Deep Midnight Blue Gradient */
    .stApp {
        background: linear-gradient(135deg, #020c1b 0%, #0a192f 100%);
        color: #ccd6f6;
    }
    
    /* Glassmorphism Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 25, 47, 0.85) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 255, 218, 0.1);
    }
    
    /* Glassmorphic Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(16, 33, 65, 0.6);
        border: 1px solid rgba(100, 255, 218, 0.2);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    /* Modern Headers */
    .dashboard-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#64ffda, #4895ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .dashboard-subheader {
        font-size: 1rem;
        color: #8892b0;
        letter-spacing: 1px;
        margin-bottom: 30px;
    }

    /* Style Info Boxes */
    .stAlert {
        background: rgba(10, 25, 47, 0.5) !important;
        border: 1px solid #64ffda !important;
        color: #64ffda !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: transparent;
        color: #64ffda;
        border: 1px solid #64ffda;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: rgba(100, 255, 218, 0.1);
        border: 1px solid #64ffda;
        box-shadow: 0 0 15px rgba(100, 255, 218, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DRAWING LOGIC (Unchanged)
# ---------------------------------------------------------
CLASS_COLORS = {
    "car": (0, 255, 0), "bus": (0, 165, 255), "truck": (0, 0, 255),
    "motorcycle": (255, 0, 255), "person": (255, 0, 0), "bicycle": (0, 255, 255)
}

def get_color(class_name):
    return CLASS_COLORS.get(class_name.lower(), (255, 255, 255))

def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        class_name = det["class_name"]
        track_id = det["tracking_id"]
        speed = det["speed_kmh"]
        color = get_color(class_name)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        id_text = f"ID:{track_id}" if track_id else "No ID"
        label = f"{class_name.capitalize()} {id_text} | {speed} km/h"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cx, cy = map(int, det["center"])
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    return frame

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.markdown('<p class="dashboard-header">💎 Edge-Optimized Real-Time Vehicle Analytics Pipeline</p>', unsafe_allow_html=True)
st.markdown('<p class="dashboard-subheader">INTELLIGENT TRAFFIC MONITORING • MICROSERVICE ARCHITECTURE</p>', unsafe_allow_html=True)

col_video, col_metrics = st.columns([3, 1])

with col_video:
    st.markdown("#### 🛰️ Live Analytics Stream")
    video_placeholder = st.empty()

with col_metrics:
    st.markdown("#### 📊 Telemetry")
    st.write("") 
    metric_fps = st.empty()
    st.write("")
    metric_latency = st.empty()
    st.write("")
    metric_objects = st.empty()
    
    st.markdown("---")
    st.markdown("#### 📜 Event Log")
    summary_placeholder = st.empty()

with st.sidebar:
    st.title("Settings")
    uploaded_file = st.file_uploader("Upload Source Video", type=["mp4", "avi", "mov"])
    
    st.markdown("---")
    try:
        res = requests.get(f"{API_URL}/")
        if res.status_code == 200:
            st.success("API STATUS: OPERATIONAL")
        else:
            st.warning("API STATUS: UNSTABLE")
    except:
        st.error("API STATUS: OFFLINE")

    st.markdown("---")
    with st.expander("🛠️ Tech Stack", expanded=True):
        st.caption("Backend: FastAPI (Uvicorn)")
        st.caption("Model: YOLO-v11n (ONNX)")
        st.caption("Logic: ByteTrack / OpenCV") 
        st.caption("Edge Deployment: NVIDIA Jetson")

# ---------------------------------------------------------
# MAIN EXECUTION LOOP (Unchanged Logic)
# ---------------------------------------------------------
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    try:
        requests.post(f"{API_URL}/reset")
    except:
        st.error("Connection to Backend Failed.")
        st.stop()

    cap = cv2.VideoCapture(tfile.name)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    start_button = st.sidebar.button("RUN PIPELINE", use_container_width=True)
    
    if start_button:
        frame_count = 0
        total_api_time = 0.0
        
        while cap.isOpened():
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret: break 
                
            frame_count += 1
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            api_start = time.time()
            try:
                response = requests.post(
                    f"{API_URL}/detect/frame",
                    files={"file": ("frame.jpg", frame_bytes, "image/jpeg")},
                    data={"fps": video_fps}
                )
                result = response.json()
            except: break
                
            api_latency = time.time() - api_start
            total_api_time += api_latency

            if result.get("success"):
                data = result["data"]
                detections = data["detections"]
                
                annotated_frame = draw_detections(frame, detections)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                loop_time = time.time() - loop_start
                e2e_fps = 1.0 / loop_time if loop_time > 0 else 0
                avg_latency_ms = (total_api_time / frame_count) * 1000
                
                metric_fps.metric(label="THROUGHPUT", value=f"{e2e_fps:.1f} FPS")
                metric_latency.metric(label="LATENCY", value=f"{avg_latency_ms:.1f} MS")
                metric_objects.metric(label="ACTIVE TRACKS", value=len(detections))
                summary_placeholder.info(data["summary"])
            else:
                break
                
        cap.release()
        st.success("Pipeline Execution Finished.")