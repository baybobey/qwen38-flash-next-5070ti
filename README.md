# qwen38-flash-next-5070ti

English | [Türkçe](README.tr.md)

Self-contained installer and tuned serving configs for **Qwen3.8-Flash-Next EXL3 4.05bpw**
on **1x or 2x 16 GB GPUs** (tuned on 2x RTX 5070 Ti), via
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) and
[theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI).

It downloads everything it needs (engine, server, model) and writes a working
262144-token-context configuration. Nothing secret lives in this repo: the API key is
generated on your machine at install time.

```
./install-2gpu.sh          # or ./install-1gpu.sh
./serve-2gpu.sh            # or ./serve-1gpu.sh
./verify.sh                # in a second terminal
```

## Hardware requirements

| | |
|---|---|
| GPUs | 1x or 2x NVIDIA, 16 GB class (tuned on Blackwell sm_120; Ampere and newer work out of the box — see below; 14 GB minimum, more VRAM = better) |
| Driver | NVIDIA >= 570 (CUDA 12.8 runtime; the installer checks) |
| System RAM | 128 GB recommended. The weights are ~100 GiB and stay resident (experts on CPU + the 36 GiB n-gram table in RAM). The installer refuses below 90 GiB |
| Disk | ~120 GB free (model 100.1 GiB + venv ~8 GB) |
| OS / Python | Linux x86_64, Python 3.10–3.13 (prebuilt engine wheel; 3.14 works via JIT, which needs the CUDA toolkit installed) |
| PCIe | Wider is faster: decode/prefill stream CPU experts over PCIe, so x8/x16 helps |
| CPU | 16+ cores recommended (CPU experts run on the host; the configs assume ~16) |

**Which GPUs work:** the prebuilt engine wheel carries compiled kernels for **sm_80, sm_86,
sm_89, sm_90, sm_100 and sm_120** — Ampere, Ada, Hopper and Blackwell (the 30/40/50-series
GeForce cards, A100/A6000, H100, B200). Turing (sm_75) and Volta (sm_70) get no kernel image
from it; the installer detects that before downloading anything, switches to the JIT build
(compiled on first import, so the CUDA toolkit is required) and says so. NVIDIA only — there
is no AMD/Intel/CPU path. Mixed-generation pairs (e.g. a 4090 next to a 5070 Ti) are fine as
long as both cards are sm_80 or newer. `./install-2gpu.sh --dry-run` prints the plan,
including which engine wheel it picked, without changing anything.

## What the installer does

Into this directory (plus the model, default `$HOME/models/Qwen3.8-Flash-Next`):

- `.venv/` — torch `2.9.0+cu128`, exllamav3 `1.5.1` (prebuilt wheel matching your
  Python; falls back to the JIT wheel + a CUDA toolkit), TabbyAPI dependencies, and the `hf` CLI
- `tabbyAPI/` — clone pinned to commit `f07131c`
- `tabbyAPI/config.yml` — from `configs/config-<N>gpu.yml`, with a `models/<name>` symlink and
  the forced sampler preset installed
- `tabbyAPI/api_tokens.yml` — a random `api_key` and `admin_key`, `chmod 600`, never printed
  and never in git
- `tabbyAPI/templates/froggeric-qwen38-v225.jinja` — froggeric's fixed Qwen chat template
  (v22.5; the file name drops the dot because tabbyAPI resolves names with `Path.with_suffix`),
  fetched from the author's own repository at a pinned revision and sha256-verified; the
  config's `prompt_template:` selects it (see “Chat template” below)
- the model — `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`, pinned to
  commit `55a732e0` (100.1 GiB, resumable; verified against the HF manifest afterwards)

Options: `--skip-model` (you already have the weights), `--model-dir D`, `--venv D`,
`--start` (start the server when done), `--dry-run` (print the plan, change nothing),
`--latest-template` (take upstream's current template file instead of the pinned revision),
`--official-template` (use Qwen's own chat template), `-y`. Full list: `./install.sh --help`.

Re-running the installer is safe and idempotent: the venv, clone and config are refreshed,
the model and the API key are kept.

## Chat template

The configs serve **froggeric's fixed Qwen chat template** instead of the one inside the
model. `install.sh` downloads it from the author's own repository at a pinned revision and
verifies its sha256 — it is deliberately **not** bundled with this repo, so both the credit
and the canonical copy stay with its author:

- Source: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (Apache-2.0)
- Installed as `tabbyAPI/templates/froggeric-qwen38-v225.jinja` and selected by
  `prompt_template: froggeric-qwen38-v225` in `config.yml`; the startup log prints
  `Using template "froggeric-qwen38-v225" for chat completions.`

Prefer Qwen's own template? Install with `--official-template`, clear `prompt_template:` in
`tabbyAPI/config.yml`, or switch at runtime with
`POST /v1/template/switch {"prompt_template_name": "…"}`.

The template is pinned by revision + sha256 so every install of this repo is reproducible. If
upstream publishes a newer revision, the installer tells you so (and which pin it used);
`--latest-template` fetches whatever upstream serves right now and prints the version stamp and
sha256 it received, so you can pin that yourself if you keep it.

## Running

```
./serve-2gpu.sh                    # 2-GPU autosplit + MTP draft
GPU_ORDER=1,0 ./serve-2gpu.sh      # pick the device order (the author's box used 1,0)
GPU_ID=1 ./serve-1gpu.sh           # single-GPU on a specific card
MODEL_DIR=/data/models/qwen ./serve-2gpu.sh
```

Stop with Ctrl-C (`pkill -f "main.py"` works too; graceful unload takes ~20 s).
The server listens on `0.0.0.0:8001` and needs the key from
`tabbyAPI/api_tokens.yml`:

```
grep api_key tabbyAPI/api_tokens.yml
```

Any OpenAI-compatible client: `base_url = http://127.0.0.1:8001/v1`,
`api_key = <that value>`, `model = Qwen3.8-Flash-Next-EXL3-405`.
`./verify.sh` checks health, `n_ctx`, and does one test completion.

The same file also holds an `admin_key` (32 hex characters, not needed for chat). It covers the
management endpoints — `/v1/model/load`, `/v1/model/unload`, `/v1/download`, the LoRA and
embedding load/unload pairs, and `/v1/template/*` + `/v1/sampling/override/*`; read it with
`grep admin_key tabbyAPI/api_tokens.yml`.

## Measured performance

Author's box: 2x RTX 5070 Ti (16 GB each, x8 + x4 PCIe), Ryzen 9 9950X3D2, 128 GiB RAM.
Sampler as configured server-side: **temp 1.0 / top-k 20 / top-p 0.95 / repetition 1.0**,
xhigh reasoning, 262144 context. Numbers below are from that machine — expect yours to
differ with PCIe width, RAM speed and CPU core count.

| Layout | Config | Prefill (33K cold) | Decode |
|---|---|---|---|
| 2 GPUs | `config-2gpu.yml` — CPU experts 416, MTP dynamic draft (8-token ceiling), chunk 2048 | ~825 t/s | 32.7–34.3 t/s stability rounds, 42.9 t/s warm repeat |
| 1 GPU | `config-1gpu.yml` — CPU experts 496, no draft, chunk 3072 | ~943 t/s | 30.5–31.3 t/s |

The 2-GPU layout wins on prefill and serves better under long prompts; the 1-GPU layout is
faster on short-context decode and leaves the second card free. Decode is CPU-expert bound:
use `cpu_moe_threads` ≈ physical cores / 2 and do not run a second model alongside.

## Tuning notes (the knobs that actually move the needle)

- `cpu_moe_split_experts` — how many experts stay in system RAM per layer (416 of 512 on
  2 GPUs, 496 of 512 on 1 GPU). Lower = more VRAM used, usually faster; raise it if you OOM.
- `cpu_moe_threads` — 16 (2-GPU) / 14 (1-GPU) on a 16-core host. Oversubscribing hurts.
- `chunk_size` — 2048 (2-GPU) / 3072 (1-GPU) here; larger speeds up prefill until the
  PLE workspace allocation OOMs mid-prefill.
- `autosplit_reserve` — headroom (MB per card) that autosplit leaves free. Keep it **small
  (96 MB) on both layouts**: exllamav3 1.5.x reserves each MoE layer's worst-case prefill
  transient (~700-750 MiB per card on 2 GPUs) inside the device budget, and the old 1024 MB
  value makes the 2-GPU load fail with "Insufficient VRAM in split for model and cache".
- `draft_mode: mtp` (2-GPU only) with `dynamic_draft: true` and an 8-token ceiling — the
  model's built-in MTP head; a real gain on 2 GPUs, nothing measurable on 1.
- `cache_mode: 6,6` — 6-bit K/V at 262144 context. Do not lower the context.
- The **forced sampler preset** (`sampler_overrides/qwen38_flash_next_thinking.yml`) is not
  optional: without it, clients that omit sampling parameters run with `top_k 0 / top_p 1.0`
  (untruncated = token soup at long context). The startup log must NOT contain the
  "No sampler override preset is configured" warning.
- `tool_format: qwen3_coder` + `reasoning: true` — the model's tool-call format and a
  reasoning parser that returns `reasoning_content` separately from `content`.

## Troubleshooting

- **"Insufficient VRAM" at load** — check `autosplit_reserve` first: it must be `[96, 96]`
  (a larger reserve cannot work with exllamav3 1.5.x on 2 GPUs, see the tuning notes).
  Then lower `cpu_moe_split_experts` by 8–16, or reduce `cache_size`. Do not reduce context
  below 262144 for this stack.
- **Prefill dies part-way with a small allocation error** — that is the PLE workspace;
  reduce `chunk_size` (e.g. 2048 → 1536).
- **First start takes minutes / kernels rebuild** — exllamav3 JIT-compiles CUDA kernels on
  first import (the installer pre-warms this). If a build is interrupted, delete
  `~/.cache/torch_extensions` and start again.
- **Gibberish or language drift** — check the startup log for the sampler warning above and
  that `api_tokens.yml` auth is actually being sent by your client.
- **Download interrupted** — re-run the installer; `hf download` resumes.

## Credits and licenses

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — MIT
- [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI) — **AGPL-3.0**. This repo clones
  it at install time rather than vendoring it; if you modify and expose it as a network service,
  the AGPL's source-availability obligations apply to you.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — Apache-2.0. The fixed chat template is **downloaded at install time, not redistributed here**
  (pinned revision `855bffc4…`, sha256 `e57684ba…`). Credit froggeric if you pass it on, and check
  upstream for newer versions.
- Model weights are **downloaded, not redistributed** here:
  [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
  quantized from Qwen's release under the Qwen community license — check the license terms
  before commercial use.

No warranty; the configs are tuned for the hardware above and will need adjustment elsewhere.
