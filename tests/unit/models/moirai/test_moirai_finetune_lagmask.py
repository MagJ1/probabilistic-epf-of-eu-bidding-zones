# tests/unit/models/moirai/test_moirai_finetune_lagmask.py
import pytest

from models.moirai.moirai_finetune_lagmask import MoiraiFinetuneLagMask
from models.moirai.custom_transforms import MaskTailOptional

from uni2ts.transform import AddObservedMask
from uni2ts.model.moirai import MoiraiModule


from collections.abc import Iterable

def _flatten_transforms(tf):
    """
    Flatten a composed Transformation chain into a list of leaf transforms.

    Works for GluonTS/Uni2TS chains created via `t1 + t2 + ...`,
    where the chain stores children in something like `.transformations`.
    """
    for attr in ("transformations", "transforms", "_transformations"):
        if hasattr(tf, attr):
            items = getattr(tf, attr)

            # IMPORTANT: ignore methods/properties that are callable
            if callable(items):
                continue

            # Only treat as a container if it's an iterable of transforms
            if isinstance(items, Iterable) and not isinstance(items, (str, bytes, dict)):
                out = []
                for t in items:
                    out.extend(_flatten_transforms(t))
                return out

    # Leaf transform (e.g., GetPatchSize, PackFields, AddObservedMask, ...)
    return [tf]


def test_train_transform_includes_masktail_with_correct_steps_and_order():
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

    ft = MoiraiFinetuneLagMask(
        module=base,
        # required by MoiraiFinetune:
        min_patches=2,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        max_dim=32,
        num_training_steps=10,
        num_warmup_steps=0,
        train_lag_steps=7,
        val_lag_steps=13,
        fill_value=0.0,
    )

    tf = ft.train_transform_map[object](
        distance=1,
        prediction_length=24,
        context_length=128,
        patch_size=32,
    )

    ts = _flatten_transforms(tf)

    masks = [t for t in ts if isinstance(t, MaskTailOptional)]
    assert masks, "MaskTailOptional not found in train transformation chain"
    assert any(m.steps == 7 for m in masks), f"Expected steps=7, got {[m.steps for m in masks]}"

    idx_mask = next(i for i, t in enumerate(ts) if isinstance(t, MaskTailOptional))
    idx_obs  = next(i for i, t in enumerate(ts) if isinstance(t, AddObservedMask))
    assert idx_mask < idx_obs, "MaskTailOptional must run BEFORE AddObservedMask"


def test_val_transform_includes_masktail_with_correct_steps_and_order():
    base = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

    ft = MoiraiFinetuneLagMask(
        module=base,
        # required by MoiraiFinetune:
        min_patches=2,
        min_mask_ratio=0.0,
        max_mask_ratio=0.0,
        max_dim=32,
        num_training_steps=10,
        num_warmup_steps=0,
        train_lag_steps=7,
        val_lag_steps=13,
        fill_value=0.0,
    )

    tf_train = ft.train_transform_map[object](
        distance=1,
        prediction_length=24,
        context_length=128,
        patch_size=32,
    )
    ts_train = _flatten_transforms(tf_train)

    tf_val = ft.val_transform_map[object](
        offset=0,
        distance=1,
        prediction_length=24,
        context_length=128,
        patch_size=32,
    )

    ts_val = _flatten_transforms(tf_val)

    masks = [t for t in ts_train if isinstance(t, MaskTailOptional)]
    assert len(masks) >= 1, "MaskTailOptional not found in train transformation chain"

    assert any(m.steps == 7 for m in masks), f"Expected 7 MaskTailOptional(steps=7), got {[m.steps for m in masks]}"

    idx_mask = next(i for i, t in enumerate(ts_train) if isinstance(t, MaskTailOptional))
    idx_obs  = next(i for i, t in enumerate(ts_train) if isinstance(t, AddObservedMask))
    assert idx_mask < idx_obs, "MaskTailOptional must run BEFORE AddObservedMask to propagate unobserved tail"


    masks = [t for t in ts_val if isinstance(t, MaskTailOptional)]
    assert len(masks) >= 1, "MaskTailOptional not found in val transformation chain"

    assert any(m.steps == 13 for m in masks), f"Expected val MaskTailOptional(steps=13), got {[m.steps for m in masks]}"

    idx_mask = next(i for i, t in enumerate(ts_val) if isinstance(t, MaskTailOptional))
    idx_obs  = next(i for i, t in enumerate(ts_val) if isinstance(t, AddObservedMask))
    assert idx_mask < idx_obs, "MaskTailOptional must run BEFORE AddObservedMask to propagate unobserved tail"