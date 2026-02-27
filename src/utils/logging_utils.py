# utils/logging_utils.py
from __future__ import annotations
import logging, os, sys
from logging.handlers import RotatingFileHandler

def init_logging(
    log_dir: str | None = None,
    run_id: str | None = None,
    level: str | int | None = None,
    fmt: str | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    rank: int = int(os.environ.get("RANK", "0")),
    coexist_with_hydra: bool = True,   # <-- key flag
    unify_format: bool = False,        # set True to push `fmt` to all handlers
):
    """
    Configure logging so it co-exists with Hydra's logging.

    - If Hydra already installed handlers (very likely under @hydra.main),
      we do NOT remove/clear them. We simply add our own file handler.
    - If there are no handlers yet (e.g., running a plain script), we
      set up console + optional file handler ourselves.
    - Optionally unify the formatter across *all* handlers.
    """
    # -------- resolve level/format ----------
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    level = getattr(logging, str(level).upper(), logging.INFO)

    if fmt is None:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    root = logging.getLogger()
    root.setLevel(level)

    # *** Do not clear existing handlers when coexisting with Hydra ***
    already_has_handlers = bool(root.handlers)

    # Console handler (only if none exist and we are sole manager)
    if not already_has_handlers or not coexist_with_hydra:
        # We take responsibility for console in this case
        if rank == 0:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter(fmt))
            root.addHandler(ch)

    # Always add our file handler if a log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fname = f"{run_id or 'run'}.log"
        fh = RotatingFileHandler(
            os.path.join(log_dir, fname), maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)

    # Optionally push the same formatter/level to all existing handlers
    if unify_format:
        formatter = logging.Formatter(fmt)
        for h in root.handlers:
            try:
                h.setLevel(level)
                h.setFormatter(formatter)
            except Exception:
                pass

    # Quiet noisy libs a bit (keeps Hydra logs intact)
    for noisy in ["urllib3", "matplotlib", "PIL", "numba"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)