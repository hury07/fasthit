"""Defines the Random explorer class."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import fasthit
from fasthit.utils import sequence_utils as s_utils


class Random(fasthit.Explorer):
    """A simple random explorer.

    Chooses a random previously measured sequence and mutates it.

    A good baseline to compare other search strategies against.

    Since random search is not data-driven, the model is only used to score
    sequences, but not to guide the search strategy.
    """

    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        seed: Optional[int] = None,
        mu: float = 1,
        elitist: bool = False,
    ):
        """
        Create a random search explorer.

        Args:
            mu: Average number of residue mutations from parent for generated sequences.
            elitist: If true, will propose the top `expmt_queries_per_round` sequences
                generated according to `model`. If false, randomly proposes
                `expmt_queries_per_round` sequences without taking model score into
                account (true random search).
            seed: Integer seed for random number generator.

        """
        name = f"Random_mu={mu}"

        super().__init__(
            name,
            encoder,
            model,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            log_file,
            seed,
        )
        self._mu = mu
        self._alphabet = alphabet
        self._elitist = elitist

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        old_sequences = measured_sequences["sequence"]
        old_sequence_set = set(old_sequences)
        new_seqs = set()
        while len(new_seqs) <= self.model_queries_per_round:
            seq = self._rng.choice(old_sequences)
            new_seq = s_utils.generate_random_mutant(
                seq,
                self._mu / len(seq),
                alphabet=list(self._alphabet),
                self._rng,
            )
            if new_seq not in old_sequence_set:
                new_seqs.add(new_seq)
        ###
        new_seqs = sorted(new_seqs)
        encodings = self.encoder.encode(new_seqs)
        preds = self.model.get_fitness(encodings)
        if self._elitist:
            idxs = np.argsort(preds)[: -self.expmt_queries_per_round-1 : -1]
        else:
            idxs = np.random.randint(0, len(new_seqs), size=self.expmt_queries_per_round)
        return measured_sequences, np.array(new_seqs)[idxs], preds[idxs]
    """
    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        eval_size = min(len(measured_sequences), self.model_queries_per_round)
        sorted_df = measured_sequences.sort_values(
            by=["true_score"], ascending=False
        ).reset_index(drop=True)
        filtered_seqs = sorted_df.loc[:eval_size-1]
        idxs = np.random.choice(eval_size, self.training_data_size, replace=False)
        sampled_seqs = filtered_seqs.loc[idxs]
        return sampled_seqs
    """
    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences
