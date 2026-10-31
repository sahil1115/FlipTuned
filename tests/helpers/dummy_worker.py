"""Dummy worker for process-control tests.

Usage: dummy_worker.py <state_dir> <mode> [--child]

- writes <state_dir>/<role>.pid with its own PID and prints a UTF-8 line;
- mode "cooperative": exits 0 once <state_dir>/cancel.requested exists;
- mode "stubborn": ignores the cancel marker and SIGTERM, and starts a grandchild that does the same.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    state = Path(sys.argv[1])
    mode = sys.argv[2]
    role = "grandchild" if "--child" in sys.argv else "worker"
    if mode == "stubborn" and hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print(f"{role} started ✓ (pid {os.getpid()})", flush=True)
    if role == "worker" and mode == "stubborn":
        subprocess.Popen([sys.executable, __file__, str(state), mode, "--child"])
    (state / f"{role}.pid").write_text(str(os.getpid()), encoding="utf-8")
    marker = state / "cancel.requested"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if mode == "cooperative" and marker.exists():
            print(f"{role} cancelled", flush=True)
            return 0
        time.sleep(0.05)
    return 1


if __name__ == "__main__":
    sys.exit(main())
