"""Defines the Machine Learning-assisted Directed Evolution (MLDE) explorer class."""
import os
import warnings
from typing import Optional, Tuple

import random
import numpy as np
import pandas as pd

import fasthit
from fasthit.utils import sequence_utils as s_utils

warnings.filterwarnings(action="ignore")
_filedir = os.path.dirname(os.path.abspath(__file__))

### TODO 1. to be generatized; 2. need more beautiful coding
class MLDE(fasthit.Explorer):
    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int, #constrained_space_size
        training_data_size: int,
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
    ):
        name = f"MLDE_model={model}"

        super().__init__(
            name,
            encoder,
            model,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            training_data_size,
            log_file=log_file,
        )

        self.alphabet = alphabet

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
        landscape: fasthit.Landscape,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        last_round = measured_sequences["round"].max()
        if last_round == 0:
            ddg_file = os.path.join(
                _filedir,
                "../landscapes/data/gb1/ddG_data.csv"
            )
            ddG_df = pd.read_csv(ddg_file)
            
            ### filtered out variants not contained in the landscape search space.
            idxs = []
            for idx in range(len(ddG_df)):
                if ddG_df.loc[idx, "AACombo"] in landscape.sequences.keys():
                    idxs.append(idx)
            ddG_df = ddG_df.loc[idxs]
            ### prioritized and filtered by ddG
            self.ddG_df = ddG_df.sort_values(by=["ddG"], ascending=True).reset_index(drop=True)
            self.search_map = self.ddG_df.loc[:self.model_queries_per_round-1]
            ### sample training data
            train_idxs = random.sample(range(len(self.search_map)), self.training_data_size)
            test_idxs = [idx for idx in range(len(self.search_map)) if idx not in train_idxs]
            train_seqs = self.search_map.loc[train_idxs]["AACombo"].to_numpy()
            ### search space shrinks
            self.search_map = self.search_map.loc[test_idxs].reset_index(drop=True)
            return measured_sequences, train_seqs, np.nan
        ###
        eval_size = self.expmt_queries_per_round - self.training_data_size
        ### search space size is smaller than eval_size,
        ### all the remaining variants will be measured
        if len(self.search_map) <= eval_size:
            new_seqs = self.search_map["AACombo"].to_numpy()
            new_fitness = landscape.get_fitness(new_seqs)
            measured_sequences = measured_sequences.append(
                pd.DataFrame(
                    {
                        "sequence": new_seqs,
                        "model_score": np.nan,
                        "true_score": new_fitness,
                        "round": last_round,
                        "model_cost": self.model.cost,
                        "measurement_cost": len(measured_sequences) + len(new_seqs),
                    }
                )
            )
            ### empty the search space
            self.search_map.drop(self.search_map.index, inplace=True)
            train_seqs = self.search_map["AACombo"].to_numpy()
            return measured_sequences, train_seqs, np.nan
        ### prioritize the remaining variants
        ### and propose top {eval_size} variants to measure
        test_seqs = self.search_map["AACombo"].to_numpy()
        encodings = self.encoder.encode(test_seqs)
        test_scores = self.model.get_fitness(encodings)
        eval_idxs = test_scores.argsort()[::-1][:eval_size]
        new_seqs = [test_seqs[idx] for idx in eval_idxs]
        new_fitness = landscape.get_fitness(new_seqs)
        measured_sequences = measured_sequences.append(
            pd.DataFrame(
                {
                    "sequence": new_seqs,
                    "model_score": np.nan,
                    "true_score": new_fitness,
                    "round": last_round,
                    "model_cost": self.model.cost,
                    "measurement_cost": len(measured_sequences) + len(new_seqs),
                }
            )
        )
        ### propose top measured fitness variant
        top_idx = new_fitness.argmax()
        top_seq = new_seqs[top_idx]
        ### shrink the search space by Combinatorial Directed Evolution
        search_pattern = len(top_seq) * ["[A-Z]"] ### TODO determined by self.alphabet
        for i in range(len(top_seq)):
            if self.starting_sequence[i] != top_seq[i]:
                search_pattern[i] = top_seq[i]
        search_pattern = "".join(search_pattern)
        mask1 = self.ddG_df["AACombo"].str.match(search_pattern)
        mask2 = self.ddG_df["AACombo"] != top_seq
        ddG_df = self.ddG_df[mask1 & mask2].reset_index(drop=True)
        ###
        test_idxs = [idx for idx in range(len(self.search_map)) if idx not in eval_idxs]
        self.search_map = self.search_map.loc[test_idxs].reset_index(drop=True)
        constrained_space_size = min(self.model_queries_per_round, len(ddG_df))
        self.search_map = ddG_df.loc[:constrained_space_size-1]
        ### search space size is small than training data size requested
        if len(self.search_map) <= self.training_data_size:
            train_seqs = self.search_map["AACombo"].to_numpy()
            self.search_map.drop(self.search_map.index, inplace=True)
            return measured_sequences, train_seqs, np.nan
        ###
        train_idxs = random.sample(range(len(self.search_map)), self.training_data_size)
        test_idxs = [idx for idx in range(len(self.search_map)) if idx not in train_idxs]
        train_seqs = self.search_map.loc[train_idxs]["AACombo"].to_numpy()
        self.search_map = self.search_map.loc[test_idxs].reset_index(drop=True)
        return measured_sequences, train_seqs, np.nan

    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        last_round = measured_sequences["round"].max()
        mask = measured_sequences["round"] == last_round
        return measured_sequences[mask]
