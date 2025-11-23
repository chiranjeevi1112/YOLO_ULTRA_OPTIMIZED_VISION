"""
Hybrid server: TCP inference pipeline + FastAPI control plane + YOLOv8 inference.

Features:
 - Real YOLO inference (ultralytics)
 - Placeholder fallback if model missing
 - Periodic FPS logging (to stdout + logs/server.log)
 - WebSocket broadcasting of inference results
 - /health and /stats (with FPS, latency, queue size, cpu/mem)
 - Robust async TCP server
"""

import asyncio
import struct
import msgpack
import time
import logging
import argparse
import socket
from asyncio import Queue
from fastapi import FastAPI, WebSocket
import uvicorn
import threading
import psutil
import numpy as np
import cv2
import os
import json

# Try to import YOLO
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False

LENGTH_PREFIX = 4

# Prepare logs directory
os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", "server.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)]
)

def pack_msg(obj: dict):
    payload = msgpack.packb(obj, use_bin_type=True)
    return struct.pack(">I", len(payload)) + payload

async def read_exact(reader, n):
    data = b""
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data += chunk
    return data

async def read_msg(reader):
    length_bytes = await read_exact(reader, LENGTH_PREFIX)
    (length,) = struct.unpack(">I", length_bytes)
    payload = await read_exact(reader, length)
    return msgpack.unpackb(payload, raw=False)

class InferenceServer:
    def __init__(self, host="0.0.0.0", port=9000, max_queue=256,
                 model_path="yolov8n.pt", device="cpu", imgsz=640):

        self.host = host
        self.port = port
        self.frame_queue = Queue(maxsize=max_queue)
        self.result_queue = Queue(maxsize=max_queue)
        self._clients = set()

        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "dropped_frames": 0,
            "last_latency_ms": 0.0,
            "fps": 0.0
        }

        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz

        self.model = None
        self.model_names = {}

        self._load_model_once()

        self._last_proc = 0
        self._last_time = time.time()

    def _load_model_once(self):
        if not _HAS_ULTRALYTICS:
            logging.warning("Ultralytics not installed → using placeholder detections.")
            return

        if not os.path.exists(self.model_path):
            logging.warning(f"Model file not found at {self.model_path} → using placeholder detections.")
            return

        try:
            logging.info(f"Loading YOLO model: {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)
            self.model_names = self.model.names

            # Warm-up
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, device=self.device, verbose=False)
            logging.info("Model warm-up done.")
        except Exception as e:
            logging.exception(f"Failed to load model: {e}")
            self.model = None

    async def inference_worker(self):
        while True:
            item = await self.frame_queue.get()
            if item is None:
                break

            frame_id = item["frame_id"]
            stream = item["stream_name"]
            ts0 = item["timestamp"]
            b = item["image"]

            # decode JPEG
            arr = np.frombuffer(b, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.frame_queue.task_done()
                continue

            detections = []

            if self.model is not None:
                try:
                    r = self.model.predict(frame, device=self.device, verbose=False)[0]
                    for b in r.boxes:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                        conf = float(b.conf[0])
                        cls = int(b.cls[0])
                        label = self.model_names.get(cls, str(cls))
                        detections.append({
                            "label": label,
                            "conf": conf,
                            "bbox": [x1, y1, x2, y2]
                        })
                except Exception as e:
                    logging.exception(f"Inference failed: {e}")
            else:
                detections = [{"label": "person", "conf": 0.88, "bbox": [10, 10, 100, 200]}]

            ts1 = time.time()
            latency = (ts1 - ts0) * 1000

            self.stats["processed_frames"] += 1
            self.stats["last_latency_ms"] = latency

            result = {
                "type": "result",
                "frame_id": frame_id,
                "stream_name": stream,
                "detections": detections,
                "latency_ms": latency,
                "timestamp": ts1
            }

            await self.result_queue.put((item["client_id"], result))
            self.frame_queue.task_done()

    async def result_dispatcher(self):
        while True:
            item = await self.result_queue.get()
            if item is None:
                break

            client_id, result = item

            # Send to TCP client
            for (cid, w) in list(self._clients):
                if cid == client_id:
                    try:
                        w.write(pack_msg(result))
                        await w.drain()
                    except:
                        pass

            # Broadcast to WebSocket clients
            for ws in list(websocket_clients):
                try:
                    asyncio.create_task(ws.send_text(json.dumps(result)))
                except:
                    try: websocket_clients.remove(ws)
                    except: pass

            self.result_queue.task_done()

    async def handle_client(self, reader, writer):
        peer = writer.get_extra_info("peername")
        cid = f"{peer}"
        logging.info(f"Client connected: {cid}")
        self._clients.add((cid, writer))

        sock = writer.get_extra_info("socket")
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            while True:
                msg = await read_msg(reader)
                if msg.get("type") != "frame":
                    continue

                self.stats["total_frames"] += 1
                msg["client_id"] = cid

                try:
                    self.frame_queue.put_nowait(msg)
                except asyncio.QueueFull:
                    self.stats["dropped_frames"] += 1
        except:
            logging.info(f"Client disconnected: {cid}")
        finally:
            self._clients = set([(c, w) for (c, w) in self._clients if c != cid])
            writer.close()
            try: await writer.wait_closed()
            except: pass

    async def periodic_logger(self):
        while True:
            await asyncio.sleep(1)
            now = time.time()
            processed = self.stats["processed_frames"]

            delta = processed - self._last_proc
            elapsed = max(1e-9, now - self._last_time)

            fps = delta / elapsed
            self.stats["fps"] = fps
            self._last_proc = processed
            self._last_time = now

            logging.info(
                f"FPS={fps:.2f} total={self.stats['total_frames']} "
                f"processed={processed} dropped={self.stats['dropped_frames']} "
                f"queue={self.frame_queue.qsize()} last_latency={self.stats['last_latency_ms']:.2f}ms"
            )

    async def start(self, workers=2):
        # Workers
        for _ in range(workers):
            asyncio.create_task(self.inference_worker())

        # Dispatcher
        asyncio.create_task(self.result_dispatcher())

        # Logger
        asyncio.create_task(self.periodic_logger())

        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        logging.info(f"TCP server listening on {self.host}:{self.port}")

        async with server:
            await server.serve_forever()


# FastAPI Control Plane
app = FastAPI()
_inference_server = None
websocket_clients = []


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    s = _inference_server.stats.copy()
    s["queue_size"] = _inference_server.frame_queue.qsize()
    s["cpu"] = psutil.cpu_percent()
    s["ram"] = psutil.virtual_memory().percent
    return s


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        try: websocket_clients.remove(ws)
        except: pass


def start_fastapi(port):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


async def main(tcp_port, api_port, workers, model, device, imgsz):
    global _inference_server
    _inference_server = InferenceServer(
        port=tcp_port,
        model_path=model,
        device=device,
        imgsz=imgsz
    )

    threading.Thread(target=start_fastapi, args=(api_port,), daemon=True).start()

    await _inference_server.start(workers=workers)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tcp-port", type=int, default=9000)
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    a = p.parse_args()

    try:
        asyncio.run(main(a.tcp_port, a.api_port, a.workers, a.model, a.device, a.imgsz))
    except KeyboardInterrupt:
        logging.info("Shutting down...")
