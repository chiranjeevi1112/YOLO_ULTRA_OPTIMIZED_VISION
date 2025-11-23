#!/usr/bin/env python3
"""
client_visualizer.py
- Captures webcam frames, sends to TCP server (same protocol).
- Receives results and overlays bounding boxes on frames using OpenCV.
- Designed to work with the server implementation provided earlier.

Usage:
    python client_visualizer.py --host 127.0.0.1 --port 9000 --stream webcam_viz --fps 10

Notes:
- This is a single-process asyncio program that performs:
  - capture & send (async-friendly with small sleeps),
  - response reader updates a dictionary of detections,
  - main loop displays latest frame and draws detections when available.
- Works best at modest FPS (5–15) to avoid display lag.
"""

import argparse
import asyncio
import struct
import msgpack
import time
import cv2
import numpy as np
import json
from collections import deque

LENGTH_PREFIX = 4

def pack_msg(obj: dict) -> bytes:
    payload = msgpack.packb(obj, use_bin_type=True)
    return struct.pack(">I", len(payload)) + payload

async def read_exact(reader: asyncio.StreamReader, n: int) -> bytes:
    data = b''
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data += chunk
    return data

async def read_msg(reader: asyncio.StreamReader):
    length_bytes = await read_exact(reader, LENGTH_PREFIX)
    (length,) = struct.unpack(">I", length_bytes)
    payload = await read_exact(reader, length)
    return msgpack.unpackb(payload, raw=False)

class DetectionStore:
    """Thread-safe-ish store for latest detections indexed by frame_id."""
    def __init__(self, maxlen=100):
        self.store = {}      # frame_id -> detections
        self.recent = deque(maxlen=maxlen)

    def update(self, frame_id, detections):
        self.store[frame_id] = detections
        self.recent.append(frame_id)
        # optionally prune old entries
        if len(self.recent) == self.recent.maxlen:
            oldest = self.recent[0]
            # keep small - no aggressive prune needed

    def get(self, frame_id):
        return self.store.get(frame_id, None)

async def response_reader(reader: asyncio.StreamReader, detections: DetectionStore):
    """Read server results and update the detection store."""
    try:
        while True:
            msg = await read_msg(reader)
            # msg expected to contain frame_id & detections
            fid = msg.get("frame_id")
            dets = msg.get("detections", [])
            if fid is not None:
                detections.update(fid, dets)
            # also print minimal info
            print(f"[result] frame={fid} dets={len(dets)} latency_ms={msg.get('latency_ms')}")
    except ConnectionError:
        print("[response_reader] connection closed")
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("response_reader error:", e)

def draw_boxes(frame, detections):
    """Draw detections on the frame. Expects bbox format [x1,y1,x2,y2]."""
    h, w = frame.shape[:2]
    for d in detections:
        bbox = d.get("bbox", [])
        conf = d.get("conf", None)
        label = d.get("label", "")
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            # clamp to int
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{label} {conf:.2f}" if conf is not None else label
            cv2.putText(frame, txt, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

async def capture_and_send(writer: asyncio.StreamWriter, stream_name: str, fps: float, shared):
    """Capture frames and send to server; store last_frame in shared dict for display."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam (device 0).")
    fid = 0
    delay = 1.0 / fps if fps and fps > 0 else 0.05
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            # store last frame for display (copy)
            shared["last_frame"] = frame.copy()
            shared["last_frame_id"] = fid

            # jpeg encode
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                await asyncio.sleep(delay)
                continue

            msg = {
                "type": "frame",
                "stream_name": stream_name,
                "frame_id": fid,
                "timestamp": time.time(),
                "image": buf.tobytes()
            }
            try:
                writer.write(pack_msg(msg))
                await writer.drain()
            except Exception as e:
                print("Error sending frame:", e)
                break

            fid += 1
            # sleep remaining time
            elapsed = time.time() - t0
            to_wait = max(0.0, delay - elapsed)
            await asyncio.sleep(to_wait)
    finally:
        cap.release()

async def main(host, port, stream_name, fps):
    reader, writer = await asyncio.open_connection(host, port)
    detections = DetectionStore()
    shared = {"last_frame": None, "last_frame_id": None}

    reader_task = asyncio.create_task(response_reader(reader, detections))
    sender_task = asyncio.create_task(capture_and_send(writer, stream_name, fps, shared))

    try:
        # display loop
        while True:
            frame = shared.get("last_frame")
            fid = shared.get("last_frame_id")
            if frame is not None:
                # overlay detections for latest frame id if available
                dets = detections.get(fid)
                if dets:
                    draw_boxes(frame, dets)
                cv2.putText(frame, f"frame_id:{fid}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.imshow("client_visualizer", frame)
            # waitKey required for imshow to work; keep around 1 ms
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        reader_task.cancel()
        sender_task.cancel()
        try:
            await reader_task
        except:
            pass
        try:
            await sender_task
        except:
            pass
        writer.close()
        await writer.wait_closed()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--stream-name", default="webcam_viz")
    p.add_argument("--fps", type=float, default=10.0)
    args = p.parse_args()
    asyncio.run(main(args.host, args.port, args.stream_name, args.fps))
