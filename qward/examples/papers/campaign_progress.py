"""Opt-in campaign diagnostics on stderr, separate from CLI JSON output."""

import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event, Thread


def report(message):
    """Emit flushed, timestamped progress when explicitly enabled."""
    if os.environ.get("CAMPAIGN_PROGRESS") == "1":
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"[{stamp}] {message}", file=sys.stderr, flush=True)


@contextmanager
def stage(label, *, interval=30):
    """Report elapsed time while an operation runs; never poll IBM from the thread.

    A heartbeat means the logging thread is alive, not that the operation is
    advancing. Completion or failure is reported separately.
    """
    if os.environ.get("CAMPAIGN_PROGRESS") != "1":
        yield
        return
    started = time.monotonic()
    stopped = Event()

    def heartbeat():
        while not stopped.wait(interval):
            report(f"{label}: still waiting for completion ({time.monotonic() - started:.0f}s)")

    report(f"{label}: started")
    worker = Thread(target=heartbeat, name="campaign-progress", daemon=True)
    worker.start()
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "failed/interrupted"
        raise
    finally:
        stopped.set()
        worker.join()
        report(f"{label}: {outcome} ({time.monotonic() - started:.1f}s)")
