import numpy as np
from scipy.stats import norm
import pandas as pd

from typing import Optional, Tuple

import fasthit
from fasthit.utils import sequence_utils as s_utils


class BO_ENU(fasthit.Explorer):
    """Explorer using unlimited Bayesian Optimization.

    IMPORTANT: This explorer is not limited by any virtual screening restriction,
    and is used to find the unrestricted performance of Bayesian Optimization
    techniques in small landscapes.

    Reference: http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """

    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int, # default: 384
        model_queries_per_round: int, # default: 800
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        util_func: str = "LCB",
        eval_batch_size: int = 256,
    ):
        """Initialize the explorer."""
        name = f"BO_ENU_Explorer-util_function={util_func}"
        assert hasattr(model, "uncertainties")
        ###
        super().__init__(
            name,
            encoder,
            model,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            log_file,
        )
        self._alphabet = alphabet
        self._best_fitness = 0.
        self._seq_len = len(starting_sequence)
        self._eval_batch_size = eval_batch_size
        util_funcs = {
            "UCB": self.UCB,
            "LCB": self.LCB,
            "TS": self.TS,
            "EI": self.EI,
            "PI": self.PI,
            "Greedy": self.Greedy,
        }
        self._util_func = util_funcs[util_func]

    def _pick_seqs(self):
        """Propose a batch of new sequences.
        """
        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            nonlocal new_seqs, maxima
            if len(curr_seq) == self._seq_len:
                new_seqs.append(curr_seq)
                if len(new_seqs) >= self._eval_batch_size:
                    encodings = self.encoder.encode(new_seqs)
                    fitness = self.model.get_fitness(encodings)
                    utility = self._util_func(fitness)
                    maxima.extend(
                        [utility[i], fitness[i], new_seqs[i]]
                        for i in range(len(new_seqs))
                    )
                    new_seqs = []
            else:
                for char in list(self._alphabet):
                    enum_and_eval(curr_seq + char)
        maxima = []
        new_seqs = []
        enum_and_eval("")
        # Sort descending based on the value.
        return sorted(maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
        landscape: Optional[fasthit.Landscape] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose `batch_size` samples."""
        samples = []
        new_fitnesses = []
        all_measured_seqs = set(measured_sequences["sequence"].values)
        for _, new_fitness, new_seq in self._pick_seqs():
            if (
                len(samples) < self.expmt_queries_per_round
                and new_seq not in all_measured_seqs
            ):
                self._best_fitness = max(new_fitness, self._best_fitness)
                samples.append(new_seq)
                all_measured_seqs.add(new_seq)
                new_fitnesses.append(new_fitness)
        return measured_sequences, np.array(samples), np.array(new_fitnesses)

    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences

    def UCB(self, preds):
        kappa = 0.05
        return preds + kappa * self.model.uncertainties

    def LCB(self, preds):
        kappa = 0.05
        return preds - kappa * self.model.uncertainties

    def TS(self, preds):
        return np.random.normal(preds, self.model.uncertainties)

    def EI(self, preds):
        eps = 0.1
        improves = preds - self._best_fitness - eps
        z = improves / self.model.uncertainties
        return improves * norm.cdf(z) + self.model.uncertainties * norm.pdf(z)

    def PI(self, preds):
        eps = 0.2
        return norm.cdf(
            (preds - self._best_fitness - eps)
            / self.model.uncertainties
        )

    def Greedy(self, preds):
        return preds
