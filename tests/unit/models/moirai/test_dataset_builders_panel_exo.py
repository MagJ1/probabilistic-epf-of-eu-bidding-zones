# tests/unit/models/moirai/test_dataset_builders_panel_exo.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import datasets
from models.moirai.simple_panel_exo import (
    _from_panel_exo_dataframe,
    SimpleDatasetBuilder,
    SimpleFinetuneDatasetBuilder,
    SimpleEvalDatasetBuilder,
    generate_finetune_builder,
    generate_eval_builder,
    generate_eval_builders,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _make_panel_exo_df(
    *,
    n_ids: int = 2,
    T: int = 10,
    freq: str = "H",
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    ck_cols: list[str] | None = None,
    cu_cols: list[str] | None = None,
    add_nans: bool = True,
) -> pd.DataFrame:
    """
    Create a small panel_exo-like DataFrame with columns:
      unique_id, ds, y, ck*, cu*
    """
    ck_cols = ck_cols or ["ck1", "ck2"]
    cu_cols = cu_cols or ["cu1"]

    rows = []
    for i in range(n_ids):
        uid = f"id{i}"
        ts = pd.date_range("2024-01-01", periods=T, freq=freq)
        y = np.arange(T, dtype=float) + i * 100.0

        ck1 = np.sin(np.arange(T))
        ck2 = np.cos(np.arange(T))
        cu1 = np.linspace(0, 1, T)

        if add_nans:
            # introduce NaNs in feature cols to exercise ffill + fillna(0.0)
            ck1 = ck1.copy()
            cu1 = cu1.copy()
            ck1[0] = np.nan     # leading NaN should become 0.0 after fill
            cu1[3] = np.nan     # internal NaN should be forward-filled

        for t_idx in range(T):
            rows.append(
                {
                    id_col: uid,
                    time_col: ts[t_idx],
                    target_col: y[t_idx],
                    "ck1": ck1[t_idx],
                    "ck2": ck2[t_idx],
                    "cu1": cu1[t_idx],
                }
            )
    return pd.DataFrame(rows)


def _collect_examples(gen_func, n: int | None = None):
    out = []
    for k, ex in enumerate(gen_func()):
        out.append(ex)
        if n is not None and (k + 1) >= n:
            break
    return out


# ---------------------------------------------------------------------
# Tests: _from_panel_exo_dataframe
# ---------------------------------------------------------------------

def test_panel_exo_basic_fields_and_shapes():
    df = _make_panel_exo_df(n_ids=2, T=12, add_nans=True)
    gen, features = _from_panel_exo_dataframe(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        ck_cols=["ck1", "ck2"],
        cu_cols=["cu1"],
        freq="H",
    )

    exs = _collect_examples(gen)
    assert len(exs) == 2

    ex0 = exs[0]
    assert set(ex0.keys()) == {
        "item_id",
        "start",
        "freq",
        "target",
        "feat_dynamic_real",
        "past_feat_dynamic_real",
    }
    assert isinstance(ex0["item_id"], str)
    assert str(ex0["freq"]) in ("H", "h")  # pandas may infer lowercase
    assert ex0["target"].dtype == np.float32
    assert ex0["target"].ndim == 1
    assert ex0["target"].shape[0] == 12

    # (features, time) shape convention here: (F, T)
    assert ex0["feat_dynamic_real"].shape == (2, 12)
    assert ex0["past_feat_dynamic_real"].shape == (1, 12)
    assert ex0["feat_dynamic_real"].dtype == np.float32
    assert ex0["past_feat_dynamic_real"].dtype == np.float32

    # NA handling:
    # - ck1[0] was NaN => after ffill and fillna(0), should be 0.0
    assert ex0["feat_dynamic_real"][0, 0] == 0.0
    # - cu1[3] was NaN => should be forward filled from cu1[2]
    assert np.isclose(ex0["past_feat_dynamic_real"][0, 3], ex0["past_feat_dynamic_real"][0, 2])

    # Features should include optional fields
    assert "feat_dynamic_real" in features
    assert "past_feat_dynamic_real" in features


def test_panel_exo_without_ck_or_cu_omits_fields():
    df = _make_panel_exo_df(n_ids=1, T=8, add_nans=False)

    # no ck / no cu
    gen, features = _from_panel_exo_dataframe(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        ck_cols=[],
        cu_cols=[],
    )
    ex = _collect_examples(gen, n=1)[0]
    assert set(ex.keys()) == {"item_id", "start", "freq", "target"}
    assert "feat_dynamic_real" not in features
    assert "past_feat_dynamic_real" not in features


def test_panel_exo_offset_trims_each_series():
    df = _make_panel_exo_df(n_ids=2, T=20, add_nans=False)
    gen, _ = _from_panel_exo_dataframe(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        ck_cols=["ck1", "ck2"],
        cu_cols=["cu1"],
        offset=5,  # keep first 5 per series
    )
    exs = _collect_examples(gen)
    assert len(exs) == 2
    assert exs[0]["target"].shape[0] == 5
    assert exs[0]["feat_dynamic_real"].shape[1] == 5


def test_panel_exo_date_offset_trims_by_time():
    df = _make_panel_exo_df(n_ids=1, T=10, add_nans=False)
    cutoff = pd.Timestamp("2024-01-01 05:00:00")
    gen, _ = _from_panel_exo_dataframe(
        df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        ck_cols=["ck1", "ck2"],
        cu_cols=["cu1"],
        date_offset=cutoff,
    )
    ex = _collect_examples(gen, n=1)[0]
    # inclusive <= date_offset; with hourly, should include 00..05 => 6 points
    assert ex["target"].shape[0] == 6


def test_panel_exo_missing_required_columns_raises():
    df = _make_panel_exo_df(n_ids=1, T=5, add_nans=False)
    df = df.drop(columns=["y"])  # remove target
    with pytest.raises(ValueError, match="Missing columns"):
        _from_panel_exo_dataframe(
            df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            ck_cols=["ck1"],
            cu_cols=["cu1"],
        )


def test_panel_exo_missing_time_col_raises():
    df = _make_panel_exo_df(n_ids=1, T=5, add_nans=False).drop(columns=["ds"])
    with pytest.raises(ValueError, match="time_col"):
        _from_panel_exo_dataframe(
            df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            ck_cols=["ck1"],
            cu_cols=["cu1"],
        )


# ---------------------------------------------------------------------
# Tests: generate_* helpers
# ---------------------------------------------------------------------

def test_generate_finetune_builder_windows_math():
    b = generate_finetune_builder(
        dataset="x",
        train_length=500,
        prediction_length=24,
        context_length=168,
        patch_size=32,
        mode="S",
        distance=2,
        storage_path=Path("/tmp"),
    )
    expected = (500 - 168 - 24) // 2 + 1
    assert b.windows == expected
    assert b.distance == 2
    assert b.prediction_length == 24
    assert b.context_length == 168
    assert b.patch_size == 32


def test_generate_eval_builder_distance_default_and_windows():
    b = generate_eval_builder(
        dataset="x",
        offset=0,
        eval_length=240,
        prediction_length=24,
        context_length=168,
        patch_size=32,
        mode="S",
        storage_path=Path("/tmp"),
        distance=None,
    )
    assert b.distance == 24
    assert b.windows == 240 // 24


def test_generate_eval_builder_custom_distance():
    b = generate_eval_builder(
        dataset="x",
        offset=0,
        eval_length=240,
        prediction_length=24,
        context_length=168,
        patch_size=32,
        mode="S",
        storage_path=Path("/tmp"),
        distance=12,
    )
    assert b.distance == 12
    assert b.windows == (240 - 24) // 12 + 1


def test_generate_eval_builders_product_size():
    builders = generate_eval_builders(
        dataset="x",
        offset=0,
        eval_length=240,
        prediction_lengths=[24, 48],
        context_lengths=[168],
        patch_sizes=[16, 32, 64],
        storage_path=Path("/tmp"),
    )
    assert len(builders) == 2 * 1 * 3
    combos = {(b.prediction_length, b.context_length, b.patch_size) for b in builders}
    assert (24, 168, 16) in combos
    assert (48, 168, 64) in combos


# ---------------------------------------------------------------------
# Tests: SimpleFinetuneDatasetBuilder.scale
# ---------------------------------------------------------------------

def test_scale_computes_mean_std_on_train_slice_and_normalizes():
    b = SimpleFinetuneDatasetBuilder(
        dataset="x",
        windows=1,
        distance=1,
        prediction_length=24,
        context_length=128,
        patch_size=32,
        mode="S",
        storage_path=Path("/tmp"),
    )
    data = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": np.arange(10, dtype=float) * 2.0,
        }
    )
    scaled = b.scale(data, start=0, end=5)
    assert b.mean is not None and b.std is not None
    train_scaled = scaled.iloc[0:5]
    # the training slice should be ~zero-mean after scaling
    assert np.allclose(train_scaled.mean().to_numpy(), 0.0, atol=1e-6)


# ---------------------------------------------------------------------
# Tests: load_dataset path selection logic (monkeypatched)
# ---------------------------------------------------------------------

def test_finetune_builder_load_dataset_selects_first_existing_path(tmp_path, monkeypatch):
    """
    Unit-test the "candidate path search" in SimpleFinetuneDatasetBuilder.load_dataset
    without invoking real FinetuneDataset logic.
    """
    ds_name = "my_ds"
    (tmp_path / ds_name).mkdir(parents=True)

    b = SimpleFinetuneDatasetBuilder(
        dataset=ds_name,
        windows=1,
        distance=1,
        prediction_length=24,
        context_length=128,
        patch_size=32,
        mode="S",
        storage_path=tmp_path,
    )

    loaded = {"path": None}

    def fake_load_from_disk(path: str):
        loaded["path"] = path
        return {"dummy": "hf_dataset"}

    # patch the datasets module that is imported inside models.moirai.simple_panel_exo
    monkeypatch.setattr(datasets, "load_from_disk", fake_load_from_disk)

    def fake_indexer(obj):
        return ("INDEXER", obj)

    def fake_finetune_dataset(windows, indexer, transform):
        return {
            "windows": windows,
            "indexer": indexer,
            "transform": transform,
        }

    # patch the symbols as imported in models.moirai.simple_panel_exo
    monkeypatch.setattr(
        "models.moirai.simple_panel_exo.HuggingFaceDatasetIndexer",
        fake_indexer,
    )
    monkeypatch.setattr(
        "models.moirai.simple_panel_exo.FinetuneDataset",
        fake_finetune_dataset,
    )

    def mk_transform(**kwargs):
        return ("TRANSFORM", kwargs)

    transform_map = {ds_name: mk_transform}

    out = b.load_dataset(transform_map=transform_map)

    assert Path(loaded["path"]) == (tmp_path / ds_name)
    assert out["windows"] == 1
    assert out["indexer"][0] == "INDEXER"
    assert out["transform"][0] == "TRANSFORM"
    assert out["transform"][1]["prediction_length"] == 24


def test_eval_builder_load_dataset_raises_if_missing(tmp_path):
    b = SimpleEvalDatasetBuilder(
        dataset="missing",
        offset=0,
        windows=1,
        distance=24,
        prediction_length=24,
        context_length=128,
        patch_size=32,
        mode="S",
        storage_path=tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        b.load_dataset(transform_map={"missing": lambda **kwargs: None})
