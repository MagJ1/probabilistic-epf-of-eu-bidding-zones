# utils/registry_utils.py
from typing import Dict, Any, List, Tuple

def exog_lists_from_registry(registry: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], List[str], List[str]]:
    """Return ordered lists from a registry: (ck, cu, static)."""
    ck, cu, static = [], [], []
    for name, meta in registry.items():  # preserves YAML insertion order (Py>=3.7)
        role = (meta.get("role") or "").lower()
        if role == "ck": ck.append(name)
        elif role == "cu": cu.append(name)
        elif role == "static": static.append(name)
    return ck, cu, static