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

pip install -r requirements.txt🚀 Running the Server

Start the server:

python server.py --tcp-port 9000 --api-port 8000 --model yolov8n.pt --device cpu --workers 2


Common flags:

--model   Path to YOLOv8 model
--device  cpu / cuda:0
--workers Number of inference workers
--imgsz   Resize dimension before inference


If no model found, server automatically uses placeholder detections.

📡 Running Clients
1️⃣ Synthetic stream client

Sends random frames:

python client.py --mode 2 --stream syn1 --fps 10 --duration 20

2️⃣ Webcam client
python client.py --mode 0 --source 0 --stream webcam_1 --fps 15

3️⃣ Video file
python client.py --mode 0 --source video.mp4 --stream vid1 --fps 10

4️⃣ Image folder
python client.py --mode 1 --folder images/ --stream img_stream --fps 5


All clients save results to:

results/<stream>.jsonl

📈 Monitoring Server Health

Open in browser:

➤ Health check
http://127.0.0.1:8000/health

➤ Full metrics
http://127.0.0.1:8000/stats


Metrics include:

total_frames

processed_frames

dropped_frames

last_latency_ms

fps

queue_size

cpu

ram

🔥 Stress Testing

Run synthetic multi-stream test:

python stress_test.py --streams 8 --fps 20 --duration 30


Use /stats to see:

Load handling

Frame drops

Server stability

Performance at scale

👁️ Live Visualizer (Webcam + YOLO Boxes)
python client_visualizer.py --host 127.0.0.1 --port 9000 --stream-name webcam_viz --fps 10


Press q to exit window.

This tool:

Sends webcam frames

Receives detections

Draws bounding boxes in real time

🗂️ Repository Structure
server.py              # Real-time inference server
client.py              # Unified client (webcam/video/images/synthetic)
client_visualizer.py   # Live bounding box viewer
stress_test.py         # Multi-client load generator
ws_demo_client.py      # Simple WebSocket JSON viewer
requirements.txt       # Dependencies
logs/                  # Server log files (auto-created)
results/               # Client JSONL output (auto-created)

🛠️ Troubleshooting
❗ Client says “Failed to open video source”

Try:

--source 1


(If you have multiple cameras)

❗ /stats not updating

Means no clients are currently sending frames.
Start a client and refresh.

❗ Placeholder bbox showing

Install ultralytics + YOLO model:

pip install ultralytics


Download yolov8n.pt into repo folder.

❗ Slow inference / overload

Try:

--imgsz 320

--workers 4

Use GPU (--device cuda:0)

✔️ Summary

This repository provides a full end-to-end real-time inference system:

Component	Purpose
server.py	Async TCP ingestion + YOLO inference + /stats + /ws
client.py	Webcam/video/images/synthetic streaming client
stress_test.py	Scalability + load testing (multi-client)
client_visualizer.py	Live bounding box overlay viewer
ws_demo_client.py	WebSocket result viewer



