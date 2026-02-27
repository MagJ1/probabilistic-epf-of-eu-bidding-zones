# utils/super_runner/trial_items.py
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

CandType = Literal["ck", "cu"]

AdoptPolicy = Literal[
    "adopt_if_improves",
    "adopt_always",
    "adopt_never",
    "adopt_if_worse",
]

QueuePolicy = Literal["pop_after_run"]  # reserved for later extensions


@dataclass(frozen=True)
class TrialPolicy:
    adopt: AdoptPolicy = "adopt_if_improves"
    queue: QueuePolicy = "pop_after_run"

    def to_dict(self) -> Dict[str, Any]:
        return {"adopt": self.adopt, "queue": self.queue}

    @staticmethod
    def from_obj(obj: Any) -> "TrialPolicy":
        """
        Accepts:
          - None -> defaults
          - dict with keys {"adopt", "queue"} (both optional)
          - string adopt policy (shorthand)
        """
        if obj is None:
            return TrialPolicy()
        if isinstance(obj, str):
            # shorthand: policy: "adopt_always"
            return TrialPolicy(adopt=obj)  # type: ignore[arg-type]
        if not isinstance(obj, dict):
            raise ValueError(f"policy must be dict/str/None, got {type(obj)}")

        adopt = obj.get("adopt")
        queue = obj.get("queue")
        return TrialPolicy(adopt=adopt, queue=queue)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Candidate:
    name: str
    type: CandType

    def label(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Candidate":
        return Candidate(name=str(d["name"]), type=str(d["type"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class CandidateGroup:
    group: tuple[Candidate, ...]

    def label(self) -> str:
        names = sorted(c.name for c in self.group)
        return "GROUP_" + "+".join(names)

    def to_dict(self) -> Dict[str, Any]:
        return {"group": [c.to_dict() for c in self.group]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CandidateGroup":
        raw = d.get("group", [])
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError(f"CandidateGroup expects non-empty 'group' list, got: {raw}")
        return CandidateGroup(group=tuple(Candidate.from_dict(x) for x in raw))


TrialItem = Union[Candidate, CandidateGroup]


def split_queue_entry(obj: Any) -> Tuple[Dict[str, Any], TrialPolicy]:
    """
    Queue entries may include a `policy` key in addition to candidate/group keys.

    Accepted shapes:
      - {"name": "...", "type": "..."}                                   (legacy)
      - {"group": [...]}                                                 (legacy)
      - {"name": "...", "type": "...", "policy": {...}}                  (new)
      - {"group": [...], "policy": {...}}                                (new)

    Returns:
      (item_dict, policy)
    where item_dict is suitable input for parse_trial_item().
    """
    if isinstance(obj, (Candidate, CandidateGroup)):
        # If someone put objects into queue, treat as legacy default policy
        return trial_item_to_dict(obj), TrialPolicy()

    if not isinstance(obj, dict):
        raise ValueError(f"Queue entry must be dict/Candidate/CandidateGroup, got {type(obj)}")

    pol = TrialPolicy.from_obj(obj.get("policy", None))

    # strip policy and keep only item-relevant keys
    if "group" in obj:
        item = {"group": obj["group"]}
    else:
        item = {"name": obj.get("name"), "type": obj.get("type")}

    return item, pol


def parse_trial_item(obj: Any) -> TrialItem:
    """
    Accepts:
      - {"name": "...", "type": "cu"}                        -> Candidate
      - {"group": [{"name": "...", "type": "cu"}, ...]}      -> CandidateGroup
      - Candidate / CandidateGroup instances                 -> pass-through
    """
    if isinstance(obj, Candidate) or isinstance(obj, CandidateGroup):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"Trial item must be dict/Candidate/CandidateGroup, got {type(obj)}")

    if "group" in obj:
        return CandidateGroup.from_dict(obj)

    if "name" not in obj or "type" not in obj:
        raise ValueError(f"Candidate expects keys {{name,type}}, got: {obj}")
    return Candidate.from_dict(obj)


def trial_item_to_dict(item: TrialItem) -> Dict[str, Any]:
    return item.to_dict()


def trial_item_label(item: TrialItem) -> str:
    return item.label()


def apply_item_to_features(base_features: Dict[str, List[str]], item: TrialItem) -> Dict[str, List[str]]:
    ck = list(base_features.get("ck_cols", []))
    cu = list(base_features.get("cu_cols", []))
    static_cols = list(base_features.get("static_cols", []))

    def add_one(c: Candidate):
        nonlocal ck, cu
        if c.type == "ck":
            if c.name not in ck:
                ck.append(c.name)
        else:
            if c.name not in cu:
                cu.append(c.name)

    if isinstance(item, CandidateGroup):
        for c in item.group:
            add_one(c)
    else:
        add_one(item)

    return {"ck_cols": ck, "cu_cols": cu, "static_cols": static_cols}


def _queue_head_matches_item(queue: List[Any], item: TrialItem) -> bool:
    if not queue:
        return False
    try:
        head_item_dict, _ = split_queue_entry(queue[0])
        head = parse_trial_item(head_item_dict)
    except Exception:
        return False
    return trial_item_to_dict(head) == trial_item_to_dict(item)