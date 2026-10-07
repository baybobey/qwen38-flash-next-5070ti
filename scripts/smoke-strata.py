#!/usr/bin/env python3
"""Functional smoke test for the Strata server: health, tool calls, multi-turn, Anthropic.

Usage: scripts/smoke-strata.py [--port 8001]
The API key is read from Strata/api_key.txt at runtime (never printed).
Prints PASS/FAIL per check; exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def http(url: str, key: str, payload=None, timeout: int = 600, extra_headers=None):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as e:
        return 0, f"connection failed: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--key-file", default=str(REPO / "Strata" / "api_key.txt"))
    a = ap.parse_args()
    key = Path(a.key_file).read_text(encoding="utf-8").strip()
    base = f"http://127.0.0.1:{a.port}"
    ok = 0
    total = 0

    def check(name, cond, detail=""):
        nonlocal ok, total
        total += 1
        ok += 1 if cond else 0
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))

    code, body = http(f"{base}/health", key)
    check("health", code == 200 and (isinstance(body, dict) and body.get("status") == "ok"), f"{code} {body}")

    tools = [{"type": "function", "function": {"name": "get_weather",
              "description": "Get current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    code, body = http(f"{base}/v1/chat/completions", key, {
        "model": "x",
        "messages": [{"role": "user", "content": "Use the get_weather tool to check Paris. Then say TOOL-OK."}],
        "tools": tools, "tool_choice": "auto", "max_tokens": 512, "temperature": 0.7})
    calls = []
    if isinstance(body, dict):
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        calls = msg.get("tool_calls") or []
    check("openai tool_calls", code == 200 and bool(calls), f"code {code}, body {str(body)[:300]}")

    code, body = http(f"{base}/v1/chat/completions", key, {
        "model": "x", "messages": [
            {"role": "user", "content": "My name is Ada."},
            {"role": "assistant", "content": "Nice to meet you, Ada!"},
            {"role": "user", "content": "What is my name? One word."}],
        "max_tokens": 512, "temperature": 0.0})
    text = ""
    if isinstance(body, dict):
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
    check("multi-turn chat", code == 200 and "Ada" in text, f"code {code}, text {text[:120]!r}")

    code, body = http(f"{base}/v1/messages", key, {
        "model": "x", "max_tokens": 512,
        "messages": [{"role": "user", "content": "Reply with exactly: ANTHROPIC-OK"}]},
        extra_headers={"anthropic-version": "2023-06-01"})
    got = ""
    if isinstance(body, dict):
        for blk in body.get("content") or []:
            got += blk.get("text") or ""
    check("anthropic endpoint", code == 200 and "ANTHROPIC" in got.upper(), f"code {code}, body {str(body)[:300]}")

    print(f"\n{ok}/{total} checks passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
