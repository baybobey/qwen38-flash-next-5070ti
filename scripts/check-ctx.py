#!/usr/bin/env python3
"""Deep-context validation for the running Strata server (262k / 500k / 1M).

Builds a ~TARGET-token prompt with a needle paragraph at a chosen depth, sends it,
verifies the needle answer, then continues the SAME conversation for a decode-at-depth
measure (this also exercises conversation reuse: the second request must read only its
tail — a full re-read means the request rendering was not consistent).

  scripts/check-ctx.py --target 512000 [--port 8001] [--needle-frac 0.9] [--label 500k]

Token counting uses Strata's own tokenizer tooling (Strata/tools),
so run it with any python that can import it; the API key is read from
Strata/api_key.txt at runtime and never printed. Raw JSON records land in results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Strata" / "tools"))

NEEDLE = ("Administrative footnote: the access code for section 42 is QILIN-7743; "
          "note it for later reference.")

FILLER_PARA = (
    "The maintenance manual for the neighborhood windmill describes a service routine that "
    "begins with a slow visual inspection of every blade, then moves to the gearbox where the "
    "technician records oil color and bearing temperature in a logbook that has been kept since "
    "the mill was commissioned. Nothing about this text needs to be remembered; it exists only "
    "to fill the context window with ordinary prose while the benchmark measures how fast the "
    "engine reads long prompts and how steadily it writes afterwards. "
)


def load_tokenizer():
    """Strata's BPE tokenizer over the pack's tokenizer files."""
    import strata_tokenizer as ST

    tdir = REPO / "Strata-data" / "packs" / "iq3_s" / "tokenizer"
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    token_type = json.loads((tdir / "token_type.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(toks, merges, token_type)


def stream(port: int, key: str, payload: dict, timeout: float = 2400.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t0 = time.monotonic()
    ttft = None
    usage = None
    timings = None
    txt: list[str] = []
    rsn: list[str] = []
    last = t0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            last = time.monotonic()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            ch = ev.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                if d.get("content"):
                    txt.append(d["content"])
                if d.get("reasoning_content"):
                    rsn.append(d["reasoning_content"])
                if ttft is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.monotonic() - t0
    total = last - t0
    out = {"text": "".join(txt), "reasoning": "".join(rsn), "usage": usage, "timings": timings,
           "ttft_s": ttft, "total_s": total}
    if usage and ttft:
        ct = usage.get("completion_tokens") or 0
        span = total - ttft
        out["decode_tps"] = (ct - 1) / span if span > 0 and ct > 1 else None
        out["read_tps"] = (usage.get("prompt_tokens") or 0) / ttft if ttft > 0 else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--target", type=int, default=512000, help="target prompt tokens")
    ap.add_argument("--needle-frac", type=float, default=0.9, help="needle position as a text fraction")
    ap.add_argument("--label", default="ctx")
    ap.add_argument("--key-file", default=str(REPO / "Strata" / "api_key.txt"))
    a = ap.parse_args()

    key = Path(a.key_file).read_text(encoding="utf-8").strip()
    tok = load_tokenizer()

    filler = FILLER_PARA * (a.target // 90 + 40)
    for _ in range(8):
        ids = tok.encode(filler, parse_special=False)
        if abs(len(ids) - a.target) / a.target < 0.012:
            break
        filler = filler[: max(2000, int(len(filler) * a.target / max(1, len(ids)) * 0.997))]
    pos = int(len(filler) * a.needle_frac)
    pos = filler.rfind(" ", 0, pos)
    body = filler[:pos] + "\n\n" + NEEDLE + "\n\n" + filler[pos:]
    text = body + ("\n\nWhen you have read all of the above, reply with exactly the access code "
                     "given for section 42, and nothing else.")
    n = len(tok.encode(text, parse_special=False))
    print(f"prompt about {n} tokens (needle at ~{int(a.needle_frac * 100)}% of the text); sending ...", flush=True)

    msgs = [{"role": "user", "content": text}]
    r = stream(a.port, key, {
        "model": "x", "messages": msgs, "max_tokens": 64, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
    })
    ok = "QILIN-7743" in (r["text"] + " " + r["reasoning"])
    print(f"read: {round(r.get('read_tps') or 0)} t/s  ttft {r['ttft_s']:.1f}s  "
          f"answer={r['text'].strip()[:80]!r}  needle={'FOUND' if ok else 'MISSING'}", flush=True)

    msgs2 = msgs + [{"role": "assistant", "content": r["text"] or "QILIN-7743"},
                    {"role": "user", "content": "Now write a detailed, well-structured explanation "
                                                "(about 300 words) of what kind of document this "
                                                "appears to be and how it is organized."}]
    r2 = stream(a.port, key, {
        "model": "x", "messages": msgs2, "max_tokens": 520, "temperature": 1.0,
        "top_k": 20, "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True},
        # same thinking setting as request 1: the template renders the assistant turn the same
        # way, so the conversation resumes from its checkpoints (flipping this = a full re-read)
        "chat_template_kwargs": {"enable_thinking": False},
    })
    det = (r2["usage"] or {}).get("prompt_tokens_details") or {}
    print(f"continue: {round(r2.get('decode_tps') or 0)} t/s  "
          f"(prompt {r2['usage'].get('prompt_tokens') if r2['usage'] else '?'} tokens, "
          f"cached {det.get('cached_tokens', '?')}, ttft {r2['ttft_s']:.2f}s)", flush=True)

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    rec = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label, "target": a.target,
           "sent_tokens": n, "needle_found": ok, "read": r, "follow": r2}
    out = out_dir / f"check-ctx-{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
