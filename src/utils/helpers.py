from pathlib import Path
import re
def _find_best_ckpt(ckpt_dir: Path) -> Path:
    """
    Prefer the numerically best 'best-*.ckpt'; fall back to 'last.ckpt'.
    Matches 'best-XX-{metric:.4f}.ckpt' written by ModelCheckpoint.
    """
    best = list(ckpt_dir.glob("best-*.ckpt"))
    if best:
        # pick the smallest metric value from filename
        def score(p: Path) -> float:
            # filename ends with ...-{metric}.ckpt ; try to parse last float
            m = re.search(r"(-?\d+(?:\.\d+)?)\.ckpt$", p.name)
            return float(m.group(1)) if m else float("inf")
        return min(best, key=score)
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")