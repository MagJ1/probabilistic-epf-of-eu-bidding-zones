# tests/unit/models/moirai/test_moirai_pretrain_lagmask.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import pytest

from uni2ts.model.moirai import MoiraiModule
from uni2ts.transform import AddObservedMask

from models.moirai.custom_transforms import MaskTailOptional
from models.moirai.moirai_pretrain_lagmask import MoiraiPretrainLagMask  


def _flatten_transforms(tf):
    """
    Best-effort flattening for composed Uni2TS/GluonTS transformations created via `t1 + t2 + ...`.
    We only need this to check presence + ordering of transforms in the chain.
    """
    for attr in ("transformations", "transforms", "_transformations"):
        if hasattr(tf, attr):
            items = getattr(tf, attr)

            # sometimes this is a method/property; ignore if callable
            if callable(items):
                continue

            if isinstance(items, Iterable) and not isinstance(items, (str, bytes, dict)):
                out = []
                for t in items:
                    out.extend(_flatten_transforms(t))
                return out

    return [tf]


def test_pretrain_transform_includes_masktail_with_correct_steps_and_order():
    """
    Structural unit test:
    - chain contains MaskTailOptional(steps=K)
    - it occurs BEFORE AddObservedMask (critical ordering for propagating unobserved tail)
    """
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

    pt = MoiraiPretrainLagMask(
        module=base,
        min_patches=2,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        max_dim=32,
        num_training_steps=10,
        num_warmup_steps=0,
        lag_mask_steps=7,
    )

    # train_transform_map is a defaultdict(factory); any key works
    tf = pt.train_transform_map[object]()  # distance/prediction_length/context_length/patch_size unused here

    ts = _flatten_transforms(tf)

    masks = [t for t in ts if isinstance(t, MaskTailOptional)]
    assert masks, "MaskTailOptional not found in pretrain transformation chain"
    assert any(m.steps == 7 for m in masks), f"Expected MaskTailOptional(steps=7), got {[m.steps for m in masks]}"

    idx_mask = next(i for i, t in enumerate(ts) if isinstance(t, MaskTailOptional))
    idx_obs = next(i for i, t in enumerate(ts) if isinstance(t, AddObservedMask))
    assert idx_mask < idx_obs, "MaskTailOptional must run BEFORE AddObservedMask"


@pytest.mark.parametrize("k", [0, 3, 14])
def test_pretrain_transform_respects_lag_mask_steps_param(k: int):
    """
    If lag_mask_steps changes, the MaskTailOptional inside the chain should reflect it.
    (Even k=0 is fine: transform becomes a no-op, but still structurally present.)
    """
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

    pt = MoiraiPretrainLagMask(
        module=base,
        min_patches=2,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        max_dim=32,
        num_training_steps=10,
        num_warmup_steps=0,
        lag_mask_steps=k,
    )

    tf = pt.train_transform_map[object]()
    ts = _flatten_transforms(tf)

    masks = [t for t in ts if isinstance(t, MaskTailOptional)]
    assert masks, "MaskTailOptional not found in chain"
    assert any(m.steps == k for m in masks), f"Expected steps={k}, got {[m.steps for m in masks]}"


def test_train_transform_map_is_defaultdict():
    # no HF module required for this check; we just want the type
    # but we must instantiate the class; use a minimal HF module to keep it consistent
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
    pt = MoiraiPretrainLagMask(
        module=base,
        min_patches=2,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        max_dim=32,
        num_training_steps=1,
        num_warmup_steps=0,
        lag_mask_steps=1,
    )
    assert isinstance(pt.train_transform_map, defaultdict)