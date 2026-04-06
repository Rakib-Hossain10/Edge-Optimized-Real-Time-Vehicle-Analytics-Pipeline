📡 VisionStream AI: Edge-Optimized Vehicle Analytics
A high-performance microservice architecture for real-time vehicle detection, tracking, and speed estimation.



🚀 Project Overview
VisionStream AI is a dual-service system built to monitor traffic flow with high precision. Unlike monolithic AI scripts, this project uses a decoupled microservice architecture to separate heavy-duty model inference from the user interface, simulating a real-world production environment where AI models run on edge gateways (like NVIDIA Jetson) while dashboards run on the cloud.

Core Capabilities:
1. Real-time Object Detection: Leverages YOLOv11n (quantized to ONNX) for robust vehicle identification.

2. Persistent Tracking: Implements ByteTrack for consistent ID assignment across frames, even during partial occlusions.

3.Velocity Analytics: Custom-built module for speed estimation ($km/h$) using scene-calibrated pixel-to-meter mapping.

4.Micro-Memory Management: Engineered to operate within a strict 512MB RAM footprint, making it ideal for low-power edge devices.


🏗️ System Architecture
The system is split into two independent services communicating via a RESTful API:

1.Inference Engine (FastAPI Backend):

-Handles frame-by-frame image processing.

-Maintains tracking state and analytics history.

-Optimization: Uses opencv-python-headless to minimize dependencies and overhead.

2. Analytics Dashboard (Streamlit Frontend):

-Provides a high-end, "Glassmorphism" dark-themed UI.

-Streams video frames as byte-encoded packets to the backend.

-Visualizes telemetry, throughput (FPS), and latency metrics.


🛰️ Edge Deployment Philosophy
Target Platform: NVIDIA Jetson (Nano / Xavier / Orin)

While this demo is deployed on https://www.google.com/search?q=Render.com, the codebase is architected specifically for Edge AI deployment:

1. Model Interoperability: Weights are exported to ONNX format, allowing for seamless conversion to NVIDIA TensorRT for hardware-accelerated inference on Jetson boards.

2. Stateless Inference: The backend handles frame-streaming, mirroring how an IP Camera (CCTV) communicates with an AI gateway.

3. Resource Throttling: Logic is optimized to prevent memory leaks and "OOM" (Out of Memory) crashes common in small-form-factor computing.


Category,Technology
1.Deep Learning: "YOLOv11n, ONNX Runtime"
2.Tracking: "ByteTrack, Centroid Tracking"
3.Backend: "FastAPI, Uvicorn, Gunicorn"
4.Frontend: "Streamlit, Custom CSS (Glassmorphism)"
5.Computer Vision: OpenCV (Headless)
Deployment: "Render, GitHub Actions (CI/CD)"


📊 Performance Benchmarks (Render Free Tier)

Average API Latency: ~250ms per frame (CPU-only).

Peak RAM Usage: 380MB / 512MB.

Throughput: 4-6 FPS (Cloud-CPU limitation; scales to 30+ FPS on NVIDIA Jetson with TensorRT).