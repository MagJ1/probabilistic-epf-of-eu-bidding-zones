# utils/feature_select.py
from typing import Dict, List, Tuple, Optional

def select_features(
    enable_registry: bool,
    registry: Dict[str, Dict],
    base_ck: Optional[List[str]] = None,
    base_cu: Optional[List[str]] = None,
    base_static: Optional[List[str]] = None,
    active_ck: Optional[List[str]] = None,
    active_cu: Optional[List[str]] = None,
    active_static: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    base_ck = list(base_ck or [])
    base_cu = list(base_cu or [])
    base_static = list(base_static or [])
    include = set(include or [])
    exclude = set(exclude or [])

    if not enable_registry or not registry:
        ck, cu, st = base_ck, base_cu, base_static
    else:
        # START from active_* if given, otherwise from base_* (so we don’t drop user-provided lists)
        ck = list(active_ck)
        cu = list(active_cu)
        st = list(active_static)

        def apply_overrides(names):
            # remove excluded
            s = set(n for n in names if n not in exclude)
            # add included if present in registry
            s |= {f for f in include if f in registry}
            # preserve registry order where possible
            ordered = [n for n in registry.keys() if n in s]
            tail = sorted(s - set(ordered))
            return ordered + tail

        ck, cu, st = apply_overrides(ck), apply_overrides(cu), apply_overrides(st)

    return ck, cu, st