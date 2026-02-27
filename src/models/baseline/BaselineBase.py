from abc import ABC, abstractmethod
from typing import Union, Literal
from pathlib import Path

import pandas as pd
from properscoring import crps_ensemble
import numpy as np


class BaselineBase(ABC):
    """Abstract base class for Baseline Models"""

    def __init__(self, path_to_train_csv: Union[str, Path], 
                 path_to_test_csv: Union[str, Path], 
                 date_col: str = "ds",
                 target_col: str = "y",
                 group_col: str = "unique_id",
                 cols_of_interest: list[str] = ["ds", "y"],):
        
        self.path_to_train_csv = Path(path_to_train_csv)
        self.path_to_test_csv = Path(path_to_test_csv)
        self.date_col = date_col
        self.target_col = target_col
        self.group_col = group_col
        self.cols_of_interest = cols_of_interest

        assert self.path_to_train_csv.exists(), f"Path {self.path_to_train_csv} does not exist."
        assert self.path_to_test_csv.exists(), f"Path {self.path_to_test_csv} does not exist."

        self.train_df = pd.read_csv(self.path_to_train_csv)
        self.test_df = pd.read_csv(self.path_to_test_csv)
        self.train_df[self.date_col] = pd.to_datetime(self.train_df[self.date_col])
        self.test_df[self.date_col] = pd.to_datetime(self.test_df[self.date_col])

        self.train_test_split_date = self.train_df[self.date_col].max()

        self.crps_scores = []

        self._init_ts_collection()

    def _init_ts_collection(self):
        self.complete_df = pd.concat([self.train_df, self.test_df], axis=0)
        self.complete_df = self.complete_df.sort_values(by=[self.group_col, self.date_col]).reset_index(drop=True)

        grouped = self.complete_df.groupby(self.group_col)
        self.ts_collection: dict[str, pd.DataFrame] = {uid: group[self.cols_of_interest].reset_index(drop=True) for uid, group in grouped}

    def _get_indexed_ts(self, uid: str) -> pd.DataFrame:
        if not hasattr(self, "index_by_date"):
            self.index_by_date = {}
        if uid not in self.index_by_date:
            self.index_by_date[uid] = self.ts_collection[uid].set_index(self.date_col)
        return self.index_by_date[uid]

    @abstractmethod
    def predict(self, uid: str, id: Union[int, str, pd.Timestamp]):
        pass

    @abstractmethod
    def evaluate_dataset(self):
        pass

    def crps_score_per_step(self, id, observations: float, forecasts: np.array) -> tuple:
        crps_score = crps_ensemble(observations, forecasts)
        self.crps_scores.append((id,crps_score))
        return id, crps_score

    def total_crps_score(self) -> float:
        self.mean_crps_score = np.mean([elem[1] for elem in self.crps_scores])
        return self.mean_crps_score
    
    def _resolve_ref_date(self, uid: str, id_or_date: Union[int, str, pd.Timestamp]) -> pd.Timestamp:
        df = self.ts_collection[uid]
        if isinstance(id_or_date, int):
            return pd.to_datetime(df.loc[id_or_date, self.date_col])
        return pd.to_datetime(id_or_date)
    
    def _get_indices_per_uid(self, mode: Literal["train", "test"]) -> dict[str, list[int]]:
        collected_indices = {}
        for uid, df in self.ts_collection.items():
            if mode == "train":
                mask = df[self.date_col] <= self.train_test_split_date
            elif mode == "test":
                mask = df[self.date_col] > self.train_test_split_date
            else:
                raise ValueError(f"Wrong type {mode}, choose either 'train' to geth indices before {self.train_test_split_date} or 'test' for indices after that date.")
            indices = df.index[mask].tolist()
            collected_indices[uid] = indices
        return collected_indices