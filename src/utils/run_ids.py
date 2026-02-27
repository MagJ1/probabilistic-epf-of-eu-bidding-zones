# utils/run_ids.py
from datetime import datetime
from pathlib import Path

def default_run_id(make_hash: str) -> str:
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{ts}-{make_hash}"

def pick_run_id(user_run_id: str | None, computed: str, runs_root: Path) -> str:
    """Use user_run_id if provided; else use computed. If exists, add -1,-2,..."""
    run_id = user_run_id or computed
    candidate = runs_root / run_id
    if user_run_id and candidate.exists():
        # If user forces a run_id and it already exists, make it unique but visible
        i = 1
        while (runs_root / f"{run_id}-{i}").exists():
            i += 1
        run_id = f"{run_id}-{i}"
    return run_id