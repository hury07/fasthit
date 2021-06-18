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
        training_data_size: int,
        starting_sequence: str,
        log_file: Optional[str] = None,
        alphabet: str = s_utils.AAS,
        mu: float = 1,
        elitist: bool = False,
        seed: Optional[int] = None,
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
            model,
            name,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            log_file,
        )
        self.mu = mu
        self.rng = np.random.default_rng(seed)
        self.alphabet = alphabet
        self.elitist = elitist

    def propose_sequences(
        self, measured_sequences: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        old_sequences = measured_sequences["sequence"]
        old_sequence_set = set(old_sequences)
        new_seqs = set()

        while len(new_seqs) <= self.model_queries_per_round:
            seq = self.rng.choice(old_sequences)
            new_seq = s_utils.generate_random_mutant(
                seq, self.mu / len(seq), alphabet=self.alphabet
            )

            if new_seq not in old_sequence_set:
                new_seqs.add(new_seq)

        new_seqs = np.array(list(new_seqs))
        encodings = self.encoder.encode(new_seqs)
        preds = self.model.get_fitness(encodings)

        if self.elitist:
            idxs = np.argsort(preds)[: -self.expmt_queries_per_round : -1]
        else:
            idxs = self.rng.integers(0, len(new_seqs), size=self.expmt_queries_per_round)

        return new_seqs[idxs], preds[idxs]
