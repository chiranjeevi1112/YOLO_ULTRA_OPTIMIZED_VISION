"""
Unified TCP client:
    mode 0 = webcam/video
    mode 1 = images folder
    mode 2 = synthetic frames

Also saves server inference results to:
    results/<stream_name>.jsonl
"""

import argparse
import asyncio
import struct
import msgpack
import time
import cv2
import numpy as np
import glob
import os
import json

LENGTH_PREFIX = 4

def pack_msg(obj):
    payload = msgpack.packb(obj, use_bin_type=True)
    return struct.pack(">I", len(payload)) + payload

async def read_exact(reader, n):
    data = b''
    while len(data) < n:
        chunk = await reader.read(n - len(data))
        if not chunk:
            raise ConnectionError("Server closed")
        data += chunk
    return data

async def read_msg(reader):
    length_bytes = await read_exact(reader, LENGTH_PREFIX)
    (length,) = struct.unpack(">I", length_bytes)
    payload = await read_exact(reader, length)
    return msgpack.unpackb(payload, raw=False)

class Saver:
    def __init__(self):
        os.makedirs("results", exist_ok=True)
        self.files = {}

    def append(self, stream, data):
        path = f"results/{stream}.jsonl"
        if stream not in self.files:
            self.files[stream] = open(path, "a", encoding="utf-8")

        self.files[stream].write(json.dumps(data) + "\n")
        self.files[stream].flush()

    def close(self):
        for f in self.files.values():
            f.close()

async def response_reader(reader, saver, save):
    try:
        while True:
            msg = await read_msg(reader)
            print("[server result]", msg)
            if save:
                saver.append(msg["stream_name"], msg)
    except:
        print("[response_reader] connection closed")

async def send_video(writer, stream, source, fps, quality, duration):
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {source}")

    start = time.time()
    fid = 0

    while True:
        if duration and time.time() - start > duration:
            break
        ret, frame = cap.read()
        if not ret:
            break

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            continue

        msg = {
            "type":"frame",
            "stream_name":stream,
            "frame_id":fid,
            "timestamp":time.time(),
            "image":buf.tobytes()
        }

        writer.write(pack_msg(msg))
        await writer.drain()

        fid += 1
        await asyncio.sleep(1.0/fps)

    cap.release()

async def send_images(writer, stream, folder, fps, quality, duration):
    files = sorted(glob.glob(folder + "/*"))
    start = time.time()
    fid = 0

    while True:
        for f in files:
            if duration and time.time() - start > duration:
                return
            frame = cv2.imread(f)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            msg = {
                "type":"frame",
                "stream_name":stream,
                "frame_id":fid,
                "timestamp":time.time(),
                "image":buf.tobytes()
            }
            writer.write(pack_msg(msg))
            await writer.drain()
            fid += 1
            await asyncio.sleep(1.0/fps)

async def send_synthetic(writer, stream, fps, quality, w, h, duration):
    start = time.time()
    fid = 0

    while True:
        if duration and time.time() - start > duration:
            break

        img = (np.random.rand(h,w,3)*255).astype("uint8")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        msg = {
            "type":"frame",
            "stream_name":stream,
            "frame_id":fid,
            "timestamp":time.time(),
            "image":buf.tobytes()
        }
        writer.write(pack_msg(msg))
        await writer.drain()
        fid += 1
        await asyncio.sleep(1.0/fps)

async def main(cfg):
    reader, writer = await asyncio.open_connection(cfg["host"], cfg["port"])
    saver = Saver()
    asyncio.create_task(response_reader(reader, saver, cfg["save"]))

    mode = cfg["mode"]
    if mode == 0:
        await send_video(writer, cfg["stream"], cfg["source"], cfg["fps"], cfg["quality"], cfg["duration"])
    elif mode == 1:
        await send_images(writer, cfg["stream"], cfg["folder"], cfg["fps"], cfg["quality"], cfg["duration"])
    elif mode == 2:
        await send_synthetic(writer, cfg["stream"], cfg["fps"], cfg["quality"], cfg["w"], cfg["h"], cfg["duration"])

    saver.close()
    writer.close()
    await writer.wait_closed()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--mode", type=int, default=2)
    p.add_argument("--stream", default="test_stream")
    p.add_argument("--source", default="0")
    p.add_argument("--folder", default="frames")
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--quality", type=int, default=80)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=480)
    p.add_argument("--save", type=bool, default=True)
    args = p.parse_args()

    cfg = vars(args)
    asyncio.run(main(cfg))
