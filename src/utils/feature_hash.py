from typing import Optional
import hashlib, json
from typing import Iterable

def feature_hash(ck_cols: Iterable[str], cu_cols: Iterable[str], static: Optional[Iterable[str]] = None) -> str:
    if static is None:
        feat = sorted(map(str, list(ck_cols) + list(cu_cols)))
    else:
        feat = sorted(map(str, list(ck_cols) + list(cu_cols) + list(static)))

    blob = json.dumps(feat, separators=(',', ':'))  # no whitespace
    return hashlib.sha1(blob.encode()).hexdigest()[:8]