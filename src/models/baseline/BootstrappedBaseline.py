from typing import Union, Optional
from tqdm import tqdm
import pandas as pd
from pandas.api.types import is_scalar
import numpy as np
from models.baseline.BaselineBase import BaselineBase

class BootstrappedBaseline(BaselineBase):
    def __init__(self, *args, 
                 point_forecast_lag: int = 7, 
                 n_samples: int,
                 ref_col: Optional[str] = None,  
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.point_forecast_lag = point_forecast_lag
        self.errors = []
        self.n_samples = n_samples
        self.ref_col = ref_col
        self._collect_errors()



    def _collect_errors(self):
        self.errors_by_uid: dict[str, list[float]] = {}

        for uid in tqdm(self.ts_collection.keys(), desc="Collecting training errors"):
            # Ensure indexed version exists
            df_indexed = self._get_indexed_ts(uid)
            train_indices = self._get_indices_per_uid(mode="train")[uid]

            valid_indices = train_indices[self.point_forecast_lag*24+1:]
            errors = []

            for i in tqdm(valid_indices, desc=f"{uid} steps", leave=False):
                try:
                    ref_date = self.ts_collection[uid].loc[i, self.date_col]
                    
                    predicted = self._make_point_prediction(uid, ref_date, self.ref_col)

                    observed_raw = df_indexed.loc[ref_date,self.target_col]

                    # to catch doubled cased because of DST
                    if not is_scalar(observed_raw):
                        observed = observed_raw.mean()
                    else:
                        observed = observed_raw

                    errors.append(observed - predicted)
                except Exception as e:
                    print(f"[{uid} @ idx {i}] Skipped due to error: {e}")
                    continue
            
            
            assert all(isinstance(x, (float, np.floating)) for x in errors)

            self.errors_by_uid[uid] = errors


    def _make_point_prediction(self, uid: str, ref_date: pd.Timestamp, ref_col: Optional[str] = None) -> float:
        col = ref_col or self.target_col

        df_indexed = self._get_indexed_ts(uid)
        # in the case of DST
        fallback_offsets = [0, -1]

        for offset in fallback_offsets:
            candidate_date = ref_date - pd.DateOffset(days=self.point_forecast_lag) + pd.Timedelta(hours=offset)
            if candidate_date in df_indexed.index:
                predicted = df_indexed.at[candidate_date, col]
                # to catch doubled cased of DST
                return predicted.mean() if not is_scalar(predicted) else predicted

        raise ValueError(f"No valid lagged date for prediction for {uid} at {ref_date}")
    
    def predict(self, uid: str, id: Union[int, str, pd.Timestamp]) -> np.array:
        ref_date = self._resolve_ref_date(uid, id)
    
        point_prediction = self._make_point_prediction(uid, ref_date, self.ref_col)

        sampled_errors = np.random.choice(self.errors_by_uid[uid], size=self.n_samples, replace=True)
        samples = point_prediction + sampled_errors

        return samples


    def evaluate_dataset(self) -> float:
        test_indices_by_uid: dict[str, list[int]] = self._get_indices_per_uid(mode="test")
        for uid, id_list in tqdm(test_indices_by_uid.items(), desc=f"UIDs", leave=False):
            for id in tqdm(id_list, desc=f"Evaluating Baseline of TS {uid}", unit="series"):
                try:
                    forecast = self.predict(uid, id)
                    true_value = self.ts_collection[uid].loc[id, self.target_col]
                    self.crps_score_per_step(id, observations=true_value, forecasts=forecast)

                except Exception as e:
                    print(f"Skipping {uid} at id={id}: {e}")

        return self.total_crps_score()

