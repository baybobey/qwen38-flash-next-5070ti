#!/usr/bin/env python3
"""Verify a local model download against the Hugging Face file manifest.

Usage:
    check_download.py <model-dir> --repo <repo-id> --revision <branch-or-sha>

Fetches the file list and sizes for the revision (path-form API URL, which is the
authoritative listing) and compares them with what is on disk, ignoring the
.cache directory. Missing files or size mismatches fail (exit 1); extra files are
reported as warnings only. Resumable downloads stage incomplete files inside
.cache, so a partial download shows up here as missing files.

Only the standard library is used.
"""

import argparse
import json
import os
import sys
import urllib.request


def fetch_manifest(repo: str, revision: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        doc = json.load(response)
    return {s["rfilename"]: s.get("size") for s in doc.get("siblings", [])}


def local_files(model_dir: str) -> dict:
    found = {}
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in files:
            path = os.path.join(root, name)
            found[os.path.relpath(path, model_dir)] = os.path.getsize(path)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"   {args.model_dir} does not exist")
        return 1

    expected = fetch_manifest(args.repo, args.revision)
    local = local_files(args.model_dir)

    missing = sorted(k for k in expected if k not in local)
    extra = sorted(k for k in local if k not in expected)
    mismatch = [
        (k, expected[k], local[k])
        for k in expected
        if k in local and expected[k] is not None and local[k] != expected[k]
    ]

    total_expected = sum(v for v in expected.values() if v)
    total_local = sum(local.values())
    print(f"   files: {len(local)} on disk, {len(expected)} in the manifest")
    print(f"   bytes: {total_local:,} on disk, {total_expected:,} expected "
          f"({total_local / 2**30:.1f} GiB)")

    for key in missing[:15]:
        print(f"   MISSING  {key}")
    if len(missing) > 15:
        print(f"   ... and {len(missing) - 15} more missing")
    for key, exp_size, got_size in mismatch[:15]:
        print(f"   SIZE     {key}: expected {exp_size:,}, found {got_size:,}")
    for key in extra[:10]:
        print(f"   extra    {key} (not in the manifest; harmless)")

    if missing or mismatch:
        print("   FAILED — re-run the download (hf download ... resumes where it stopped)")
        return 1

    print("   OK — the download matches the manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
