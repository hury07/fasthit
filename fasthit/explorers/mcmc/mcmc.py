"""Defines the MCMC explorer class."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from bisect import bisect_left

import fasthit
from fasthit.utils import sequence_utils as s_utils


class MCMC(fasthit.Explorer):
    """
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
        seed: Optional[int] = 0,
        mu: Optional[float] = 1.2,
        temperature: float = 0.06,
        batch_size: int = 10,
    ):
        """
        """
        name = f"MCSA={mu}"

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
        self._state = s_utils.string_to_one_hot(starting_sequence, alphabet)
        self._seq_len = len(self.starting_sequence)

        self._mu = mu
        self._temperature = temperature
        self._batch_size = batch_size

    def init_states(self, state):
        actions = set()
        pos_changes = []
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))
        while len(actions) < self._batch_size:
            action = []
            m = self._rng.poisson(1)
            m = max(1, m)
            m = min(m, self._seq_len)
            for pos in self._rng.choice(range(self._seq_len), m, replace=False):
                pos_tuple = tuple(self._rng.choice(pos_changes[pos]))
                action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
        actions = list(actions)
        states = []
        states_str = []
        for i in range(self._batch_size):
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for action in actions[i]:
                x[action] = 1
            state = s_utils.construct_mutant_from_sample(x, state)
            states.append(state)
            states_str.append(s_utils.one_hot_to_string(state, self._alphabet))
        fitness, uncertainty = self.fitness(states)
        return states, states_str, fitness, uncertainty

    def mutate(self, states):
        new_states = []
        for state in states:
            pos_changes = []
            for pos in range(self._seq_len):
                pos_changes.append([])
                for res in range(len(self._alphabet)):
                    if state[pos, res] == 0:
                        pos_changes[pos].append((pos, res))
            action = []
            m = self._rng.poisson(self._rng.uniform(1, self._mu))
            m = max(1, m)
            m = min(m, self._seq_len)
            for pos in self._rng.choice(range(self._seq_len), m, replace=False):
                pos_tuple = tuple(self._rng.choice(pos_changes[pos]))
                action.append(pos_tuple)
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for act in action:
                x[act] = 1
            new_states.append(s_utils.construct_mutant_from_sample(x, state))
        return new_states

    def fitness(self, states):
        states_str = []
        for state in states:
            states_str.append(s_utils.one_hot_to_string(state, self._alphabet))
        states_encoding = self.encoder.encode(states_str)
        preds = self.model.get_fitness(states_encoding)
        uncertainties = self.model.uncertainties
        return preds, uncertainties

    def update_states(self, states, fitness, new_states, init_uncertainty):
        new_fitness, new_uncertainty = self.fitness(new_states)
        y_star = np.where(new_uncertainty > 2.0 * init_uncertainty, -np.inf, new_fitness)
        ###
        prob_margin = np.minimum(1.0, np.exp((y_star - fitness)/self._temperature))
        for i in range(self._batch_size):
            if self._rng.random() < prob_margin[i]:
                states[i] = new_states[i]
                fitness[i] = new_fitness[i]
        seqs = []
        for state in states:
            seqs.append(s_utils.one_hot_to_string(state, self._alphabet))
        return states, seqs, fitness

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        last_round = measured_sequences["round"].max()
        ### set state from previous batch by Thompson sampling
        last_batch = measured_sequences[
            measured_sequences["round"] == last_round
        ]
        measured_batch = last_batch[["true_score", "sequence"]]
        measured_batch = measured_batch.sort_values(by=["true_score"])
        sampled_seq = self.Thompson_sample(measured_batch)
        self._state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
        
        ###
        samples = []
        preds = []
        prev_cost = self.model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())

        states, seqs, fitness, init_uncertainty = self.init_states(self._state)
        for seq, pred in zip(seqs, fitness):
            if seq not in all_measured_seqs:
                all_measured_seqs.add(seq)
                samples.append(seq)
                preds.append(pred)
        
        while self.model.cost - prev_cost < self.model_queries_per_round:
            new_states = self.mutate(states)
            states, seqs, fitness = self.update_states(states, fitness, new_states, init_uncertainty)
            for seq, pred in zip(seqs, fitness):
                if seq not in all_measured_seqs:
                    all_measured_seqs.add(seq)
                    samples.append(seq)
                    preds.append(pred)

        if len(samples) < self.expmt_queries_per_round:
            random_sequences = set()
            while len(random_sequences) < self.expmt_queries_per_round - len(samples):
                # TODO: redundant random samples
                random_sequences.update(
                    s_utils.generate_random_sequences(
                        self._seq_len,
                        self.expmt_queries_per_round - len(samples) - len(random_sequences),
                        list(self._alphabet),
                        self._rng,
                    )
                )
            random_sequences = sorted(random_sequences)
            samples.extend(random_sequences)
            encodings = self.encoder.encode(random_sequences)
            preds.extend(self.model.get_fitness(encodings))
        samples = np.array(samples)
        preds = np.array(preds)
        sorted_order = np.argsort(preds)[: -self.expmt_queries_per_round-1 : -1]
        return measured_sequences, samples[sorted_order], preds[sorted_order]

    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences

    def Thompson_sample(self, measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x) for x in measured_batch["true_score"].tolist()])
        fitnesses = fitnesses / fitnesses[-1]
        x = self._rng.uniform()
        index = bisect_left(fitnesses, x)
        return measured_batch.loc[index, "sequence"]