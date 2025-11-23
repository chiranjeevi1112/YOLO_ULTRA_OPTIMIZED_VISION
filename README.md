
# Ultra-Optimized Real-Time Vision Streaming System

A fast, scalable real-time inference system with:

Async TCP server (msgpack frames)

YOLOv8 inference (CPU/GPU)

FastAPI /health + /stats

Unified client (webcam/video/images/synthetic)

Stress test (multi-client)

Visualizer (live bounding boxes)

# install
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
pip install ultralytics

# Run Server
python server.py --tcp-port 9000 --api-port 8000 --model yolov8n.pt --device cpu --workers 2

# Run Clients
## Synthetic
python client.py --mode 2 --stream syn1 --fps 10

## Webcam
python client.py --mode 0 --source 0 --stream cam1 --fps 15

## Video
python client.py --mode 0 --source video.mp4 --stream vid1

## Images
python client.py --mode 1 --folder images/ --stream img1

Results saved to:

results/<stream>.jsonl

# Monitor 
http://127.0.0.1:8000/health
http://127.0.0.1:8000/stats

# stresstest
python stress_test.py --streams 8 --fps 20 --duration 30
# vizuavilizer 
python client_visualizer.py --host 127.0.0.1 --port 9000 --stream-name viz --fps 10



