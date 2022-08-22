"""Defines the Adalead explorer class."""
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import fasthit
from fasthit.utils import sequence_utils as s_utils


class Adalead(fasthit.Explorer):
    """
    Adalead explorer.

    Algorithm works as follows:
        Initialize set of top sequences whose fitnesses are at least
            (1 - threshold) of the maximum fitness so far
        While we can still make model queries in this batch
            Recombine top sequences and append to parents
            Rollout from parents and append to mutants.

    """

    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int,  # default: 384
        model_queries_per_round: int,  # default: 400
        starting_sequence: str,
        log_file: Optional[str] = None,
        alphabet: str = s_utils.AAS,
        seed: Optional[int] = 0,
        mu: int = 1,
        recomb_rate: float = 0,
        threshold: float = 0.05,
        rho: int = 0,
        eval_batch_size: int = 20,
    ):
        """
        Args:
            mu: Expected number of mutations to the full sequence (mu/L per position).
            recomb_rate: The probability of a crossover at any position in a sequence.
            threshold: At each round only sequences with fitness above
                (1-threshold)*f_max are retained as parents for generating next set of
                sequences.
            rho: The expected number of recombination partners for each recombinant.
            eval_batch_size: For code optimization; size of batches sent to model.

        """
        name = f"Adalead_mu={mu}_threshold={threshold}"

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
        self._alphabet = alphabet
        self._threshold = threshold
        self._recomb_rate = recomb_rate
        self._mu = mu  # number of mutations per *sequence*.
        self._rho = rho
        self._eval_batch_size = eval_batch_size

        self._sequences: Dict[str, float] = {}

    def _recombine_population(self, gen):
        # If only one member of population, can't do any recombining
        if len(gen) == 1:
            return gen

        self._rng.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA: List[str] = []
            strB: List[str] = []
            switch = False
            for ind in range(len(gen[i])):
                if self._rng.rand() < self._recomb_rate:
                    switch = not switch
                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])
            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret

    def _random_fill_children(self, nodes, measured_sequence_set: Set[str]):
        child_idxs: List[int] = []
        children: List[str] = []
        while len(children) < len(nodes):
            idx, node = nodes[len(children) - 1]
            # random generation
            child = s_utils.generate_random_mutant(
                node,
                self._mu / len(node),
                list(self._alphabet),
                self._rng,
            )
            # Stop when we generate new child that has never been seen
            # before
            if (
                child not in measured_sequence_set
                and child not in self._sequences
            ):
                child_idxs.append(idx)
                children.append(child)
        return children, child_idxs

    def _generate_seqs(self, roots: np.ndarray, previous_model_cost: int, measured_sequence_set: Set[str]):
        encodings = self.encoder.encode(roots)
        root_fitnesses = self.model.get_fitness(encodings)
        nodes = list(enumerate(roots))
        while (
            len(nodes) > 0
            and self.model.cost - previous_model_cost + self._eval_batch_size
            < self.model_queries_per_round
        ):
            children, child_idxs = self._random_fill_children(
                nodes, measured_sequence_set)
            ###
            encodings = self.encoder.encode(children)
            fitnesses = self.model.get_fitness(encodings)
            self._sequences.update(zip(children, fitnesses))
            nodes = [(idx, child) for idx, child, fitness in zip(
                child_idxs, children, fitnesses) if fitness >= root_fitnesses[idx]]

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        measured_sequence_set: Set[str] = set(measured_sequences["sequence"])
        # Get all sequences within `self._threshold` percentile of the top_fitness
        top_fitness = measured_sequences["true_score"].max()
        top_inds = measured_sequences["true_score"] >= top_fitness * (
            1 - np.sign(top_fitness) * self._threshold
        )
        parents = np.resize(
            measured_sequences["sequence"][top_inds].to_numpy(),
            self.expmt_queries_per_round,
        )
        self._sequences.clear()  # ordered in python>=3.7, and vice versa.
        previous_model_cost = self.model.cost
        while self.model.cost - previous_model_cost < self.model_queries_per_round:
            # generate recombinant mutants
            for i in range(self._rho):
                parents = self._recombine_population(parents)
            for i in range(0, len(parents), self._eval_batch_size):
                # Here we do rollouts from each parent (root of rollout tree)
                roots = parents[i: i + self._eval_batch_size]
                self._generate_seqs(
                    roots, previous_model_cost, measured_sequence_set)

        if len(self._sequences) == 0:
            raise ValueError(
                "No sequences generated. If `model_queries_per_round` is small, try "
                "making `eval_batch_size` smaller"
            )
        # We propose the top `self.expmt_queries_per_round` new sequences we have generated
        new_seqs = np.array(list(self._sequences.keys()))
        preds = np.array(list(self._sequences.values()))
        sorted_order = np.argsort(preds)[: -self.expmt_queries_per_round-1: -1]
        return measured_sequences, new_seqs[sorted_order], preds[sorted_order]

    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences
