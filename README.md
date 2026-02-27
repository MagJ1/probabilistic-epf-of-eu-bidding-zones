# Probabilistic Electricity Price Forecasting of European Bidding Zones

Research code for probabilistic time-series forecasting in energy markets. The repository includes model training, calibration diagnostics, scoring-rule evaluation, and statistical comparisons. Experiments are organized as modular pipelines configured via Hydra.

## Important Information

The NHITS + QRA model contains the option to calculate the Energy Score, which simplifies to the CRPS. This is, because the NHITS + QRA model currently only forecasts one horizon after the other and not combined (i.e. all 24 steps at once). 

## Basic Setup

Some model stacks require mutually exclusive dependency versions (notably **Moirai/uni2ts** vs **Chronos/ChronosX**). For that reason, install each stack in its **own virtual environment**.

Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .my_venv
source .my_venv/bin/activate
python -m pip install -U pip
```

Install the repository plus one model stack (example: **ChronosX**):

```bash
python -m pip install -e ".[chronosx]"
```

Available extras (install one per venv):

- `normalizing_flows`  — Normalizing Flow models
- `nhits_qra`          — NHITS + QRA
- `moirai`             — Moirai / uni2ts
- `chronosx`           — Chronos/ChronosX stack
- `dev`                — development tools (pytest/ruff/pre-commit, etc.)

Example (Moirai):

```bash
python -m pip install -e ".[moirai]"
```


## Data setup

### 1) DE-LU (Germany–Luxembourg)

The DE-LU dataset lives in `raw_data/single_bid_zones/de_lu/`. In addition to timestamps and prices, it also contains calendar features as well as market, trading, and generation-related features.

Use `raw_data/prep_de_lu_dataset.py` to:

- split `de_lu_train_val.csv` into **train** and **validation** (cutoff: `2022-12-31 23:00`, inclusive for train),
- optionally create **lagged features** (e.g., `load_lag7d`, `cross_border_trading_lag7d`) that serve as simple “dummy forecasts” / proxies for future-unknown covariates.

After running the script, you should have (at least):

- `de_lu_train.csv`
- `de_lu_val.csv`
- (optional) lag-augmented variants depending on your configuration.

---

### 2) Cross-border dataset (other EU bidding zones)

For the remaining bidding zones — **AT, BE, CH, CZ, DK1, DK2, FR, HU, IT-North, NL, NO2, PL, SE4, SI** — use:

- `data_crawler/crawler_bidding_zones.py`

The crawler downloads day-ahead price data from **Energy-Charts** (Fraunhofer ISE) and enriches it with calendar features. It also builds the transfer-learning datasets used in the DE-LU experiments:

- `cross_border_electricity_prices_zero_shot.csv`
- `cross_border_electricity_prices_one_shot.csv`
- `cross_border_electricity_prices_few_shot.csv`

Data source: Energy-Charts (Fraunhofer ISE), https://www.energy-charts.info

---

### 3) Next step: training

Once the datasets are prepared, you can start training the four model stacks:

- Normalizing Flows
- NHITS + QRA
- Moirai
- Chronos / ChronosX


---

## How To

### Normalizing Flows, NHITS + QRA, ChronosX

Execute the runners in each of the model folders, e.g.

```bash
python -m models.nhits_qra.cli.train_nhits_qra
```
You can include different parameters through Hydra.

---
### Moirai

Moirai (via `uni2ts`) expects datasets in **Arrow** format. The workflow is:

1. **Preprocess** a CSV dataset into an Arrow dataset.
2. **Finetune** a Moirai model using Hydra configs.

Both steps are described below.

---

### 1) Preprocess dataset

Dataset preprocessing is handled by:

- `src/models/moirai/build_moirai_dataset.py`

This project primarily uses the **`panel_exo`** format:

- One CSV contains multiple time series.
- Each series is identified by an `unique_id` column.
- Targets and covariates live in separate columns (panel structure).
- For `panel_exo`, you can specify column names via `--id_col`, `--time_col`, `--ck_cols`, `--cu_cols`.

Run:

```bash
python -m models.moirai.build_moirai_dataset \
<dataset_name> <file_path> \
--dataset_type <wide, long, wide_multivariate, panel_exo> \
--offset <int> \
--date_offset <date> \
--freq <H> \
--normalize \
--id_col <str> \
--time_col <str> \
--ck_cols [str,...] \
--cu_cols [str,...]
```

##### Example

```bash
python -m models.moirai.build_moirai_dataset \
de_lu_train \
raw_data/single_bid_zones/DE_LU/de_lu_train.csv \
--dataset_type panel_exo 
```

---

### 2) Finetune

Finetuning is configured through Hydra using:

- `src/models/moirai/conf/finetune/config_finetune.yaml`

You must set three config groups:

1. **Model**: choose from  
   `src/models/moirai/conf/finetune/model/`
2. **Train data**: choose from  
   `src/models/moirai/conf/finetune/data/`
3. **Validation data**: choose from  
   `src/models/moirai/conf/finetune/val_data/`

You can also add your own YAML configs to these folders.

Run:

```bash
python -m models.moirai.train_moirai \
  model=<model_name> \
  data=<train_data_name> \
  val_data=<val_data_name>
```

##### Example
```bash
python -m models.moirai.train_moirai \
  model=moirai_1.1_R_tiny_lagmask \
  data=de_lu_train \
  val_data=de_lu_val
```
---

## Testing

Install a dev venv, e.g. for normalizing flows as
```bash
python -m venv .dev_venv
source .dev_venv/bin/activate
pip install -e ".[dev, normalizing_flows]"
```

