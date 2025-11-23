🚀 Ultra-Optimized Real-Time Vision Streaming System

A fast, scalable, low-latency real-time vision inference system with:

Async TCP frame ingestion (msgpack + length prefix)

YOLOv8 inference (CPU/GPU, multi-worker)

FastAPI control plane (/health, /stats)

Unified client (webcam, video, images, synthetic)

Stress testing (multi-client simulation)

Live visualizer (bounding box overlay)

📦 Installation
python -m venv .venv
.venv\Scripts\activate      # Windows
# OR
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
pip install ultralytics     # if using YOLO models


Download yolov8n.pt → place in project folder.

🖥️ Run the Server
python server.py --tcp-port 9000 --api-port 8000 --model yolov8n.pt --device cpu --workers 2


Common flags:

--model <path>

--device cpu/cuda:0

--workers <num>

--imgsz <size>

If the model is missing → server uses placeholder detections.

📡 Run Clients
1️⃣ Synthetic
python client.py --mode 2 --stream syn1 --fps 10 --duration 20

2️⃣ Webcam
python client.py --mode 0 --source 0 --stream webcam_1 --fps 15

3️⃣ Video file
python client.py --mode 0 --source video.mp4 --stream vid1 --fps 10

4️⃣ Image folder
python client.py --mode 1 --folder images/ --stream img_stream --fps 5


Results saved to:

results/<stream>.jsonl

📈 Monitor the Server

Health:

http://127.0.0.1:8000/health


System metrics:

http://127.0.0.1:8000/stats


Shows:

total/processed/dropped frames

fps, latency

queue size

CPU + RAM usage

🔥 Stress Testing

Simulate multiple clients:

python stress_test.py --streams 8 --fps 20 --duration 30


Use /stats to monitor server stability and throughput.

👁️ Live Visualizer
python client_visualizer.py --host 127.0.0.1 --port 9000 --stream-name webcam_viz --fps 10


Shows:

Webcam feed

YOLO bounding boxes

Press q to exit

🗂️ Project Structure
server.py              # Inference server
client.py              # Streaming client
client_visualizer.py   # Live bounding-box viewer
stress_test.py         # Multi-client load tester
requirements.txt       # Library list
logs/                  # Server logs (auto)
results/               # JSONL outputs (auto)
