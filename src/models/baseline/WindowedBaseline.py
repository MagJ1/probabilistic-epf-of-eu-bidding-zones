from typing import Union, Optional, Literal
import pandas as pd
from tqdm import tqdm
from models.baseline.BaselineBase import BaselineBase
import numpy as np
from pathlib import Path
from properscoring import crps_ensemble


class WindowedBaseline(BaselineBase):
    def __init__(self, 
                 *args, 
                 last_days: int = 7, 
                 seasonal_months: int = 12, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_last_days = last_days
        self.n_seasonal_months = seasonal_months
    
    def _collect_values(self, uid: str, ref_date: pd.Timestamp):

        df = self.ts_collection[uid]

        min_required_date = ref_date - pd.DateOffset(months=self.n_seasonal_months)
        if df[self.date_col].min() > min_required_date:
            raise ValueError(f"Not enough history for {uid} at {ref_date}")
        
        df_indexed = self._get_indexed_ts(uid)

        last_7d = []
        n_days_into_past = 1

        while len(last_7d) < self.n_last_days:
            past_date = ref_date - pd.Timedelta(days=n_days_into_past)
            if past_date in df_indexed.index:
                row = df_indexed.loc[[past_date]]
                last_7d.append(row)
            
            n_days_into_past+=1

        last_7d = pd.concat(last_7d)
        last_7d = last_7d[~last_7d.index.duplicated(keep="first")]

        same_days = []
        n_months_into_past = 1

        while len(same_days) < self.n_seasonal_months:
            past_date = ref_date - pd.DateOffset(months=n_months_into_past)
            if past_date in df_indexed.index:
                row = df_indexed.loc[[past_date]]
                same_days.append(row)

            n_months_into_past+=1

        same_days_df = pd.concat(same_days) if same_days else pd.DataFrame(columns=df.columns)
        same_days_df = same_days_df[~same_days_df.index.duplicated(keep="first")]

        assert len(last_7d) == self.n_last_days, f"Expected {self.n_last_days} rows, got {len(last_7d)}"
        assert len(same_days_df) == self.n_seasonal_months, f"Expected {self.n_seasonal_months} rows, got {len(same_days_df)}"

        return last_7d, same_days_df
    
    def predict(self, uid: str, id: Union[int, str, pd.Timestamp]):
        ref_date = self._resolve_ref_date(uid, id)
        last_7d, same_days_df = self._collect_values(uid, ref_date)

        forecast = pd.concat([df for df in [last_7d, same_days_df] if not df.empty])

        return forecast
    
    def evaluate_dataset(self):
        test_indices_by_uid: dict[str, list[int]] = self._get_indices_per_uid(mode="test")

        for uid, id_list in tqdm(test_indices_by_uid.items(), desc=f"UIDs"):
            for id in tqdm(id_list, desc=f"Evaluating Baseline of TS {uid}", unit="series", leave=False):
                try:
                    forecast = self.predict(uid, id)
                    true_value = self.ts_collection[uid].loc[id, self.target_col]
                    self.crps_score_per_step(id, observations=true_value, forecasts=forecast[self.target_col])
                except Exception as e:
                    print(f"Skipping {uid} at id={id}: {e}")

        return self.total_crps_score()
    

    def _iter_test_anchors(
        self,
        uid: str,
        *,
        anchor_hour: int | None,
        origin_stride_hours: int,
    ) -> list[pd.Timestamp]:
        """
        Return anchor timestamps on the *test* split.
        Stride only applies if anchor_hour is not None.
        24h -> step=1 (no change), 48h -> step=2, 72h -> step=3, ...
        """
        df = self.ts_collection[uid]
        df_test = df[df[self.date_col] > self.train_test_split_date]
        if df_test.empty:
            return []

        times = pd.to_datetime(df_test[self.date_col].values)
        if anchor_hour is None:
            # No anchor -> every timestamp could be an origin, but we only want anchors.
            return []

        anchors = times[[t.hour == anchor_hour for t in times]]
        step = max(1, origin_stride_hours // 24)  # 24->1, 48->2, 72->3...
        if step > 1:
            anchors = anchors[::step]
        return anchors.tolist()


    def evaluate_day_ahead(
        self,
        *,
        anchor_hour: int = 0,
        origin_stride_hours: int = 24,
        min_points: int = 23,             # accept 23/24/25 values
        save_to: str | Path | None = None,
        offset_steps: int = 168,
    ) -> pd.DataFrame:
        """
        For each anchor t0, slice [t0, t1) where t1 is the next anchor,
        compute CRPS for every row in that slice (duplicates kept), and
        output mean per origin: unique_id | origin_ds | crps_mean
        """
        rows = []
        for uid in self.ts_collection.keys():
            idx_df = self._get_indexed_ts(uid)
            idx = idx_df.index

            t_cutoff = self.train_test_split_date + pd.Timedelta(hours=max(0, offset_steps))

            # collect all anchors from the *full* index to find the next boundary robustly
            all_anchors = pd.Index([t for t in idx if t.hour == anchor_hour]).unique().sort_values()
            # test origins = anchors strictly after train/test split
            test_origins = [t for t in all_anchors if t > t_cutoff]

            # apply stride on anchors
            step = max(1, origin_stride_hours // 24)
            if step > 1:
                test_origins = test_origins[::step]

            # map each origin to its next anchor boundary
            # (use the next element in all_anchors, not only test_origins)
            all_anchor_list = list(all_anchors)
            anchor_pos = {t: i for i, t in enumerate(all_anchor_list)}

            for t0 in test_origins:
                i0 = anchor_pos.get(t0, None)
                if i0 is None or i0 + 1 >= len(all_anchor_list):
                    # no closing boundary -> skip last incomplete day
                    continue
                t1 = all_anchor_list[i0 + 1]

                # slice [t0, t1) — keeps duplicates naturally
                mask = (idx >= t0) & (idx < t1)
                day_df = idx_df.loc[mask]
                if day_df.empty or len(day_df) < min_points:
                    continue

                crps_vals = []
                # iterate row-wise (preserves duplicates). index may repeat timestamps.
                for ts_ref, row in day_df.iterrows():
                    obs = float(row[self.target_col])

                    forecast = self.predict(uid, ts_ref)
                    if isinstance(forecast, pd.DataFrame):
                        ens = forecast[self.target_col].to_numpy()
                    else:
                        ens = np.asarray(forecast).reshape(-1)

                    if ens.size == 0:
                        continue

                    crps_vals.append(float(crps_ensemble(obs, ens)))

                if len(crps_vals) >= min_points:
                    rows.append(
                        {"unique_id": uid, "origin_ds": pd.to_datetime(t0), "crps_mean": float(np.mean(crps_vals))}
                    )

        out = pd.DataFrame(rows).sort_values(["unique_id", "origin_ds"]).reset_index(drop=True)
        if save_to is not None:
            save_to = Path(save_to); save_to.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(save_to, index=False)
        return out
    
    def _get_indexed_ts(self, uid: str) -> pd.DataFrame:
        if not hasattr(self, "index_by_date"):
            self.index_by_date = {}
        if uid not in self.index_by_date:
            df = self.ts_collection[uid].sort_values(self.date_col).copy()
            # keep ALL rows (duplicates allowed) -> slicing between anchors will handle them
            self.index_by_date[uid] = df.set_index(self.date_col)
        return self.index_by_date[uid]
    

    

