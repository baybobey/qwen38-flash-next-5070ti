# qwen38-flash-next-5070ti — Strata edition

English | [Türkçe](README.tr.md)

Self-contained installer and tuned serving configs for **Qwen3.8-Flash-Next GSQ-RCO IQ3_S**
on **1x or 2x 16 GB GPUs** (tuned on 2x RTX 5070 Ti), via
[Niko1221/Strata](https://github.com/Niko1221/Strata).

It downloads everything it needs (engine source, model, MTP draft layer, chat template),
builds the CUDA engine locally, and writes a working 262,144-token configuration with the
tuning this box measured. Nothing secret lives in this repo: the API key is generated on
your machine at install time.

The **exllamav3 + TabbyAPI stack** (EXL3 4.05, the slower but higher-precision route)
lives on the [`exllamav3`](https://github.com/baybobey/qwen38-flash-next-5070ti/tree/exllamav3)
and [`latest`](https://github.com/baybobey/qwen38-flash-next-5070ti/tree/latest) branches.

```
./install-strata-2gpu.sh         # or ./install-strata-1gpu.sh
./serve-2gpu-strata.sh           # or ./serve-1gpu-strata.sh
./verify-strata.sh                # in a second terminal
```

## Hardware requirements

| | |
|---|---|
| GPUs | 1x or 2x NVIDIA, 16 GB class (tuned on Blackwell sm_120; the engine compiles for your arch) |
| Driver | NVIDIA >= 580 with CUDA 13 for RTX 50 / sm_120 builds (CUDA >= 12.4 covers sm_86+) |
| Build tools | `cmake`, `ninja`, `nvcc` — Strata ships no prebuilt Linux engine; the installer compiles it (~2 min on 16 cores) |
| System RAM | 128 GB recommended. The GGUF keeps ~47 GiB of experts + the 26.9 GiB n-gram (PLE) table resident; the installer refuses below ~62 GiB |
| Disk | ~120 GB free (model 83.6 GB + prepared pack ~1.5 GB + MTP draft ~5 GB + engine build) |
| OS / Python | Linux x86_64, Python 3.10+ (Strata's own setup creates the venv) |
| PCIe | Wider is faster: decode streams uncached experts over PCIe, so x8/x16 helps |
| CPU | 16+ cores recommended (CPU expert pools + the prompt stager) |

## What the installer does

Into this directory:

- `Strata/` — the Strata clone pinned to commit `e8ca9af` (v0.1.40.2), then its own
  `./setup.sh --yes --family qwen --model IQ3_S --context 262144 --kv int8 --vision no
  --no-start --build`: venv, CUDA engine (compiled for your GPU), model download,
  the prepared pack (`Strata-data/packs/iq3_s`) and the MTP draft layer (`Strata-data/mtp`,
  fetched from Qwen's BF16 checkpoint by Strata's own `tools/mtp_fetch.py` — ~5 GB of
  range reads; the 360 GB checkpoint is never downloaded)
- `Strata/api_key.txt` — a random key, `chmod 600`, never printed, never in git
- `Strata/strata-iq3_s.json` — the run config, written from `configs/config-<N>gpu.json`
  with this machine's paths resolved and the key injected (git-ignored)
- the chat template — froggeric's fixed Qwen template (`froggeric-v22.5`), fetched from
  the author's repo at a pinned revision, sha256-verified, with one local
  patch: `_default_reasoning_effort` pinned `medium` → `xhigh`. Installed into
  the pack's tokenizer dir (Strata prefers `<tokenizer>/chat_template.jinja` over
  its built-in rendering); the stock template is kept as `chat_template.orig.jinja.bak`
- `Strata/strata-iq3_s.shared-settings.json` — `reasoning_effort: high` for clients
  that send none (git-ignored)

Options: `--skip-model` (weights already downloaded), `--model-dir D`, `--data-dir D`,
`--calibrate` (~10 min per-GPU-layout measurement of `--pcie-frac` / `--spec-min-p` —
recommended on any box that is not the author's), `--context N`
(262144 | 524288 | 1048576), `--port N`, `--start`, `--dry-run`, `-y`. Full list:
`./install-strata.sh --help`.

Re-running is safe and idempotent: the clone, engine, venv, pack, configs and template are
refreshed; the model and the API key are kept.

## Running

```
./serve-2gpu-strata.sh                     # 2 GPUs, layer split auto, 262k (the tuned default)
./serve-1gpu-strata.sh                     # 1 GPU
CTX=500k ./serve-2gpu-strata.sh            # 524,288 window (experimental yarn scaling)
CTX=1m   ./serve-2gpu-strata.sh            # 1,048,576 window (experimental; ~95 GiB RAM)
GPU_IDS=1,0 ./serve-strata.sh --gpus 2     # which cards (nvidia-smi numbering)
MODEL_DIR=/data/models ./serve-2gpu-strata.sh   # a GGUF location other than the default
```

Each `serve-*` re-applies the canonical config (paths + key) and then runs the
server in the foreground; stop with Ctrl-C or `./stop-strata.sh` from another terminal.
The server listens on `0.0.0.0:8001` (keyed — keep it behind a firewall if that
matters on your network) and needs the key from `Strata/api_key.txt`:

```
cat Strata/api_key.txt
```

Any OpenAI-compatible client: `base_url = http://127.0.0.1:8001/v1`,
`api_key = <that value>`, `model = qwen3.8-flash-next-iq3_s`. There is also an
Anthropic-compatible `/v1/messages` endpoint and a built-in web chat UI at
`http://127.0.0.1:8001/`. `./verify-strata.sh` checks health, context length, and runs
the 4-part functional suite (`scripts/smoke-strata.py`: health, OpenAI tool calls,
multi-turn, Anthropic).

## Measured performance

Author's box: 2x RTX 5070 Ti (16 GB each, x8 + x4 PCIe), Ryzen 9 9950X3D2, 123 GiB RAM,
engine 0.1.40.2 built locally for sm_120. Sampler as configured server-side:
**temp 1.0 / top-k 20 / top-p 0.95 / repetition 1.0**, xhigh reasoning. Numbers are medians
of 2–4 runs with a 58.7k-token prompt unless noted — expect yours to differ with PCIe
width, RAM speed and CPU core count (`--calibrate` re-measures the engine's own knobs).

| Profile | Prefill (58.7k cold) | Decode 2k-out | Decode @58.7k depth |
|---|---|---|---|
| 2 GPUs, `--prefill 12288 --pipeline-windows 2` | ~4,290 t/s | ~111–134 t/s | ~106–110 t/s |
| 1 GPU, chunk 12288 (borrows) | ~3,370 t/s | ~88–90 t/s | ~75 t/s |
| 258k full-window prompt (98.4% of 262k) | 4,843 t/s engine (53 s) | — | ~130 t/s sample |

Context variants (experimental; `scripts/check-ctx.py` validates end-to-end — a needle
at 90% depth found, then continuation at depth):

| Window | Read | Decode at depth | Notes |
|---|---|---|---|
| 524,288 (yarn 2) | 4,187 t/s @ 509,861 tok | ~99–109 t/s | `--kv-resident 98304`; parking 12 GiB |
| 1,048,576 (yarn 4) | 3,462–3,499 t/s @ 1,016,182 tok | ~71–84 t/s | keep `--kv-resident 65536` (see borrow cap) |

For comparison, the exllamav3 stack on the same box: 825–943 t/s prefill, 30–34 t/s
decode. IQ3_S is a ~3.5-bit quant vs EXL3 4.05 bpw — Strata is the fast lane, the
exllamav3 branch keeps the higher-precision one.

## Tuning notes (the knobs that actually move the needle)

- `--prefill 12288` + prompt-cache borrowing — the measured knee on 0.1.39+
  (4096→2.2k, 6144→3.0k, 8192→3.4k, 12288→4.3k t/s; 16384 buys +1% for a 92% transient
  borrow of a card's cache). Do **not** add `--no-prefill-borrow` on a split: own
  prompt buffers shrink the expert cache (3,880 vs 5,222 slots/card) and cost ~10% deep
  decode.
- `--pipeline-windows 2` (2-GPU split only): +15–20% decode; costs 272 MiB/card.
- `--kv int8 --kv-resident 65536`: the streaming KV ring. All-in-VRAM (`--kv-resident 0`)
  costs 18–25% decode here — the expert cache beats KV residency in this engine.
- `--ple-io ram`: the 26.9 GiB n-gram table resident (falls back from `mlock` to
  page pre-touch when `ulimit -l` is smaller; that is fine).
- conversation parking (`--conversation-cache-mib 8192 --conversation-cache-slots 4`):
  a disjoint request (another chat, a client's auxiliary calls) parks the live
  conversation to RAM instead of wiping the checkpoint chain; returning costs ~0.4 s
  instead of a full re-prefill (~30 s at 48k tokens). Works with `--layer-split` since
  0.1.39.
- the `"sampling"` block in the config (temp 1.0 / top-k 20 / top-p 0.95):
  Strata applies these as **defaults** — a client that sends its own values still wins.
  Without them, clients that omit sampling run untruncated (token soup at long context).
- `env.STRATA_IQ_MT_MIN=1`: reproducible greedy output with IQ packs (upstream #152) —
  costs ~1–3% decode; remove to revert.
- calibrated per GPU layout: `--pcie-frac 0.20 / --spec-min-p 0.70` (2-GPU), `0.35 / 0.70`
  (1-GPU) on the author's box — re-measure yours with `--calibrate` or
  `cd Strata && ./setup.sh --calibrate --no-start`.

## Context variants and the borrow cap

The trained window is 262,144. The 500k/1m profiles extend it with
`--rope-scaling yarn --rope-scale F` (F = window / 262144) — **experimental**: the rope
change applies at every position, so a scaled run is a slightly different model even
inside 262k; keep each window in its own config and never mix turns across windows.

The trap to know: a prompt chunk runs at its configured size only if its
per-card borrow (~0.27 slots/token: 12288 → ~3,310) fits the smallest expert cache minus
128 slots. Otherwise the engine logs `prompt chunk X -> Y`, silently downgrades, and
reads drop 20–30%. `--kv-resident`, `--pipeline-windows` and the window size all move
the caches — after any change, grep the engine log (`Strata/strata-iq3_s.log`) for
`prompt chunk`.

## Updating the engine

`cd Strata && git fetch --tags && git checkout <new tag> && ./update.sh` — then re-run
`./setup.sh --calibrate --no-start` (the measured optima move with the engine kernels)
and re-check the `prompt chunk` line at your context. The configs here track what was
last verified on the author's box; the current pin is engine v0.1.40.2 (`e8ca9af`).

## Troubleshooting

- **reads much slower than the table** — check for a `prompt chunk X -> Y` downgrade in
  `Strata/strata-iq3_s.log` (the borrow cap), and re-run `--calibrate` for your layout.
- **"VRAM free … LOW" warning at start** — raise `--vram-reserve-mib` (1400 (2-GPU) /
  1000 (1-GPU) is what this tuning uses).
- **engine died / 503 "the engine is starting"** — the server restarts it; a kernel or
  driver reset on the box can take the desktop session down while the engine keeps
  serving (the author's AMD-iGPU box does this; run heavy jobs from a TTY/SSH).
- **first answer of a new long chat is slow, next ones fast** — working as designed: the
  engine reads the whole prompt once, checkpoints it every 16k tokens, and reuses it per
  turn. A disjoint request now *parks* instead of wiping (see parking).
- **download interrupted** — re-run the installer; the HF download and the MTP fetch
  resume where they stopped.
- **different tokens on repeated greedy output** — set `STRATA_IQ_MT_MIN=1`
  (it is in the shipped configs; engines read env at spawn, so restart after config edits).

## Credits and licenses

- [Niko1221/Strata](https://github.com/Niko1221/Strata) — **MIT**. Cloned at install time,
  not vendored; the engine is compiled from that source on your machine.
- Model weights are **downloaded, not redistributed** here:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  (revision `ed59f920…`, both shards sha-verified by Strata's setup), quantized from
  Qwen's release under the Qwen community license — check the license terms before
  commercial use. The MTP draft layer is range-fetched from Qwen's
  [BF16 checkpoint](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Strata's tooling.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — Apache-2.0. **Downloaded at install time, not redistributed here** (pinned
  revision `855bffc4…`, sha256 `e57684ba…`); credit froggeric if you pass it on.

No warranty; the configs are tuned for the hardware above and will need adjustment elsewhere.
