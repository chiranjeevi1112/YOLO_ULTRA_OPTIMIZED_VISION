# Ultra-Optimized Real-Time Vision Streaming System

This repository implements a **high-performance real-time vision inference system** with a hybrid architecture:

- **TCP Data Plane** → streams raw frames using msgpack  
- **FastAPI Control Plane** → provides health, metrics, and WebSocket events  
- **Async YOLO Inference Pipeline** → CPU/GPU, multi-worker  
- **Unified Client Tools** → webcam/video/image-folder/synthetic  
- **Stress Testing System** → simulate multiple cameras  
- **Live Visualizer** → real-time bounding box overlay  

This project demonstrates scalable, low-latency, multi-stream deep-learning inference.

---

# ✨ Features

### ✔ Asynchronous TCP Frame Server  
- Processes frames from any number of clients  
- Msgpack + length-prefix protocol  
- Frame queue + backpressure handling  
- Per-stream processing isolation  

### ✔ YOLOv8 Inference (CPU/GPU)  
- Loads model once  
- Warm-up pass for faster first inference  
- Supports custom model paths  
- Fallback “dummy detection” if no model installed  

### ✔ FastAPI Control Plane  
Endpoints:
- `/health` → server alive  
- `/stats` → processed frames, dropped frames, FPS, latency, CPU/RAM, queue size  
- `/ws` → WebSocket live detection broadcast  

### ✔ Unified Streaming Client  
Supports:
- **Webcam**  
- **Video file**  
- **Image folder playback**  
- **Synthetic frames (random noise)**  

Saves server results to:

### ✔ Stress Test Framework  
Simulates many parallel clients to measure:
- Throughput  
- Frame drops  
- Latency stability  
- Worker saturation  

### ✔ Live Visualizer  
- Webcam → Server → Detections → OpenCV window  
- Draws bounding boxes and labels  
- Useful for demos and validation  

---

# 📦 Installation

Create virtual environment (recommended):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
