#!/usr/bin/env python3
"""End-to-end image test for the vision path (needs a profile whose config carries the "vision" block).

Generates a small labelled PNG locally (no binary in git, no download), sends it
twice as an OpenAI image_url: the first request encodes it (strata-vision), the
second reuses the cached embeddings (the server caches by image hash). Checks the
server advertises images on /health and that the answer mentions what is drawn.

Usage: scripts/test-vision.py [--port 8001] [--keep PNG_PATH]
The API key is read from Strata/api_key.txt at runtime (never printed).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def make_png(path: Path) -> None:
    """A 256x256 RGB image: black background, a filled red circle at the centre.
    Pure stdlib (zlib + struct) so the test needs no imaging package."""
    w = h = 256
    rows = bytearray()
    r, g, cx, cy = 90, 128, 128, 128
    for y in range(h):
        row = bytearray(b"\x00")                       # filter type 0
        for x in range(w):
            in_circle = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
            row += bytes((220, 30, 30) if in_circle else (10, 10, 10))
        rows += row
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


def health(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=15) as r:
        return json.loads(r.read())


def call(port: int, key: str, img_url: str, question: str, max_tokens: int = 160):
    payload = {
        "model": "x",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},  # a thinking model + small max_tokens => content null
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": question},
            ],
        }],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    msg = d["choices"][0]["message"].get("content") or ""
    return dt, d.get("usage", {}), msg.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--key-file", default=str(REPO / "Strata" / "api_key.txt"))
    ap.add_argument("--keep", metavar="PNG_PATH", help="also write the generated test image here")
    a = ap.parse_args()

    h = health(a.port)
    if not h.get("images"):
        print("FAIL: /health says images are off (the run config has no vision block, or the server "
              "started without --vision) — run ./install-strata-<N>gpu.sh and ./serve-<N>gpu-strata.sh again")
        return 1
    print(f"/health: images on, model {h.get('model')}")

    key = Path(a.key_file).read_text(encoding="utf-8").strip()
    tmp = Path("/tmp") / "strata-test-vision.png"
    make_png(tmp)
    if a.keep:
        Path(a.keep).parent.mkdir(parents=True, exist_ok=True)
        Path(a.keep).write_bytes(tmp.read_bytes())
    print(f"test image: {tmp} ({tmp.stat().st_size} bytes, 256x256: a red circle on black)")

    q = "What shape and colour is in this image? Answer in one short sentence."
    dt, u, ans = call(a.port, key, str(tmp), q)
    print(f"[send 1] {dt:6.2f}s prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
    print(f"  answer: {ans[:300]}")
    dt2, u2, ans2 = call(a.port, key, str(tmp), q)
    print(f"[send 2] {dt2:6.2f}s prompt={u2.get('prompt_tokens')} completion={u2.get('completion_tokens')}"
          "  (same picture: embeddings cached)")
    print(f"  answer: {ans2[:300]}")
    tmp.unlink(missing_ok=True)

    low = (ans + " " + ans2).lower()
    ok = ("circle" in low or "round" in low) and ("red" in low)
    print("PASS: the picture was read (circle + red in the answers)" if ok else
          "WARN: the request went through but the answer does not mention the expected shape/colour — check the model output")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
