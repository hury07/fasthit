import numpy as np
from scipy.stats import norm
import pandas as pd

from bisect import bisect_left
from typing import Optional, Tuple

import fasthit
from fasthit.utils import sequence_utils as s_utils


class BO_EVO(fasthit.Explorer):
    """
    Evolutionary Bayesian Optimization explorer.

    Algorithm works as follows:
        for N experiment rounds
            recombine samples from previous batch if it exists and measure them,
                otherwise skip
            Thompson sample starting sequence for new batch
            while less than B samples in batch
                Generate `model_queries_per_round/expmt_queries_per_round` samples
                If variance of ensemble models is above twice that of the starting
                    sequence
                Thompson sample another starting sequence
    """

    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int, # default: 384
        model_queries_per_round: int, # default: 3200
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        util_func: str = "LCB",
        recomb_rate: float = 0.,
    ):
        """
        Args:
            util_func (equal to EI or UCB): The improvement util_function used in BO,
                default UCB.
            recomb_rate: The recombination rate on the previous batch before
                BO proposes samples, default 0.

        """
        name = f"BO_EVO_proposal-function={util_func}"
        assert hasattr(model, "uncertainties")

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
        self._recomb_rate = recomb_rate
        self._best_fitness = -np.inf
        self._state = s_utils.string_to_one_hot(starting_sequence, alphabet)
        self._seq_len = len(self.starting_sequence)
        util_funcs = {
            "UCB": self.UCB,
            "LCB": self.LCB,
            "TS": self.TS,
            "EI": self.EI,
            "PI": self.PI,
            "Greedy": self.Greedy,
        }
        self._util_func = util_funcs[util_func]

    def _recombine_population(self, gen):
        np.random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if np.random.random() < self._recomb_rate:
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

    def sample_actions(self):
        """ Regularized evolution.
            Sample n actions resulting in n sequences to screen.
        """
        actions = set()
        pos_changes = []
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if self._state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))
        while len(actions) < self.model_queries_per_round / self.expmt_queries_per_round:
            action = []
            for pos in range(self._seq_len):
                if np.random.random() < 1. / self._seq_len:
                    pos_tuple = pos_changes[pos][
                        np.random.randint(len(self._alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
        return list(actions)

    def pick_action(self):
        """ Optimize utility function.
            Select 1 candidate from n candidates
        """
        state = self._state.copy()
        actions = self.sample_actions() # Regularized evolution
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.model_queries_per_round // self.expmt_queries_per_round):
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = s_utils.construct_mutant_from_sample(x, state)
            states_to_screen.append(s_utils.one_hot_to_string(state_to_screen, self._alphabet))
        encodings = self.encoder.encode(states_to_screen)
        preds = self.model.get_fitness(encodings)
        ### Optimize utility function
        action_idx = np.argmax(self._util_func(preds)) ### greedy TODO: any improvement?
        uncertainty = self.model.uncertainties[action_idx]
        ###
        action = actions_to_screen[action_idx]
        new_state_string = states_to_screen[action_idx]
        self._state = s_utils.string_to_one_hot(new_state_string, self._alphabet)
        ###
        return uncertainty, new_state_string, preds[action_idx]

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
        _last_batch_seqs = last_batch["sequence"].tolist()
        _last_batch_true_scores = last_batch["true_score"].tolist()
        last_batch_seqs = _last_batch_seqs
        if self._recomb_rate > 0 and len(last_batch) > 1:
            last_batch_seqs = self._recombine_population(last_batch_seqs)
        measured_batch = []
        _new_seqs = []
        for seq in last_batch_seqs:
            if seq in _last_batch_seqs:
                measured_batch.append(
                    (_last_batch_true_scores[_last_batch_seqs.index(seq)], seq)
                )
            else:
                _new_seqs.append(seq)
        if len(_new_seqs) > 0:
            encodings = self.encoder.encode(_new_seqs)
            fitnesses = self.model.get_fitness(encodings)
            measured_batch.extend(
                [(fitnesses[i], _new_seqs[i]) for i in range(fitnesses.shape[0])]
            )
        measured_batch = sorted(measured_batch)
        sampled_seq = self.Thompson_sample(measured_batch)
        self._state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
        
        ###
        ### generate next batch by picking actions
        initial_uncertainty = None
        samples = []
        preds = []
        prev_cost = self.model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        while self.model.cost - prev_cost < self.model_queries_per_round:
            ### Optimize utility function
            uncertainty, new_state_string, pred = self.pick_action()
            ###
            if new_state_string not in all_measured_seqs:
                self._best_fitness = max(self._best_fitness, pred)
                all_measured_seqs.add(new_state_string)
                samples.append(new_state_string)
                preds.append(pred)
            if initial_uncertainty is None:
                initial_uncertainty = uncertainty
            if uncertainty > 2. * initial_uncertainty:
                # reset sequence to starting sequence
                # if we're in territory that's too uncharted
                sampled_seq = self.Thompson_sample(measured_batch)
                self._state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
                initial_uncertainty = None
        if len(samples) < self.expmt_queries_per_round:
            random_sequences = set()
            while len(random_sequences) < self.expmt_queries_per_round - len(samples):
                # TODO: redundant random samples
                random_sequences.update(
                    s_utils.generate_random_sequences(
                        self._seq_len,
                        self.expmt_queries_per_round - len(samples) - len(random_sequences),
                        list(self._alphabet)
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

    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]

    def UCB(self, preds):
        kappa = 0.05
        return preds + kappa * self.model.uncertainties

    def LCB(self, preds):
        kappa = 0.1
        return preds - kappa * self.model.uncertainties

    def TS(self, preds):
        return np.random.normal(preds, self.model.uncertainties)

    def EI(self, preds):
        eps = 0.1
        improves = preds - self._best_fitness - eps
        z = improves / self.model.uncertainties
        return improves * norm.cdf(z) + self.model.uncertainties * norm.pdf(z)

    def PI(self, preds):
        eps = 0.1
        return norm.cdf(
            (preds - self._best_fitness - eps)
            / self.model.uncertainties
        )

    def Greedy(self, preds):
        return preds
