"""
Stress test:
Launch N synthetic clients in parallel to load-test the inference server.

Example:
    python stress_test.py --streams 8 --fps 20 --duration 30
"""

import subprocess
import argparse
import sys
import time
import os

SCRIPT = os.path.join(os.path.dirname(__file__), "client.py")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--duration", type=int, default=20)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    a = p.parse_args()

    procs = []

    for i in range(a.streams):
        cmd = [
            sys.executable, SCRIPT,
            "--mode", "2",
            "--stream", f"synthetic_{i}",
            "--fps", str(a.fps),
            "--duration", str(a.duration),
            "--host", a.host,
            "--port", str(a.port),
            "--save", "False"
        ]
        p = subprocess.Popen(cmd)
        procs.append(p)

    try:
        t0 = time.time()
        while time.time() - t0 < a.duration:
            time.sleep(1)
    finally:
        for p in procs:
            try:
                p.kill()
            except:
                pass
