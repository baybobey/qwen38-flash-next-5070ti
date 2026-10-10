#!/usr/bin/env python3
"""Stop the Strata server serving on a port (default 8001) and wait for its engine child.

Kills only the listener found via `ss -tlnp` (no pattern matching that
could hit the invoking shell), then waits for the port and the engine to disappear.
Pass --quiet for a silent no-op when nothing is running.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

QUIET = "--quiet" in sys.argv[1:]
args = [a for a in sys.argv[1:] if a != "--quiet"]
PORT = int(args[0]) if args else 8001


def listener_pid(port: int) -> int | None:
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line and "LISTEN" in line and "pid=" in line:
            return int(line.split("pid=")[1].split(",")[0])
    return None


def engine_pids() -> list[int]:
    # [e] keeps the pattern from matching the pgrep wrapper itself. The vision
    # encoder child (engine/strata-vision) is excluded on purpose — it is not a
    # `strata --serve` process and dies with its own parent.
    out = subprocess.run(["pgrep", "-f", "engin[e]/strata --serve"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main() -> int:
    pid = listener_pid(PORT)
    if pid is None:
        if not QUIET:
            print(f"no listener on :{PORT}")
        return 0
    print(f"SIGTERM -> server pid {pid} (:{PORT})")
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(1)
    else:
        print("server did not exit after 90 s; leaving it alone for a human look")
        return 1
    for _ in range(30):
        if not engine_pids():
            break
        time.sleep(1)
    else:
        print(f"engine still running: {engine_pids()}")
        return 1
    print("stopped: port free, engine gone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
