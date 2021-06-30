"""BO explorer."""
from bisect import bisect_left
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import fasthit
from fasthit.utils import sequence_utils as s_utils
from fasthit.utils.replay_buffers import PrioritizedReplayBuffer


class BO_EVO(fasthit.Explorer):
    """
    Evolutionary Bayesian Optimization (Evo_BO) explorer.

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
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        training_data_size: int,
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        proposal_func: str = "UCB",
        recomb_rate: float = 0.,
    ):
        """
        Args:
            proposal_func (equal to EI or UCB): The improvement proposal_function used in BO,
                default UCB.
            recomb_rate: The recombination rate on the previous batch before
                BO proposes samples, default 0.

        """
        name = f"BO_EVO_proposal-function={proposal_func}"
        assert hasattr(model, "uncertainties")

        super().__init__(
            name,
            encoder,
            model,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            training_data_size,
            log_file,
        )
        self._alphabet = alphabet
        self._recomb_rate = recomb_rate
        self._best_fitness = 0.0
        self._num_actions = 0
        proposal_funcs = {
            "Greedy": self._Greedy,
            "UCB": self._UCB,
            "Thompson": self._Thompson,
            "EI": self._EI,
        }
        self._proposal_func = proposal_funcs[proposal_func]

    def initialize_data_structures(self):
        """Initialize."""
        self.state = s_utils.string_to_one_hot(self.starting_sequence, self._alphabet)
        self._seq_len = len(self.starting_sequence)
        # use PER buffer, same as in DQN
        self.memory = PrioritizedReplayBuffer(
            len(self._alphabet) * self._seq_len, 100000, self.expmt_queries_per_round, 0.6
        )

    def train_models(self):
        """Train the model."""
        if len(self.memory) >= self.expmt_queries_per_round:
            batch = self.memory.sample_batch()
        else:
            self.memory.batch_size = len(self.memory)
            batch = self.memory.sample_batch()
            self.memory.batch_size = self.expmt_queries_per_round
        states = batch["next_obs"]
        state_seqs = [
            s_utils.one_hot_to_string(state.reshape((-1, len(self._alphabet))), self._alphabet)
            for state in states
        ]
        rewards = batch["rews"]
        encodings = self.encoder.encode(state_seqs)
        self.model.train(encodings, rewards)

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
        """Sample actions resulting in sequences to screen."""
        actions = set()
        pos_changes = []
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if self.state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))
        while len(actions) < self.model_queries_per_round / self.expmt_queries_per_round:
            action = []
            for pos in range(self._seq_len):
                if np.random.random() < 1 / self._seq_len:
                    pos_tuple = pos_changes[pos][
                        np.random.randint(len(self._alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
        return list(actions)

    def pick_action(self, all_measured_seqs):
        """Pick action."""
        ### select 1 from n candidates
        state = self.state.copy()
        actions = self.sample_actions()
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
        ###
        action_idx = np.argmax(self._proposal_func(preds)) ### greedy TODO: any improvement?
        uncertainty = self.model.uncertainties[action_idx]
        ###
        action = actions_to_screen[action_idx]
        new_state_string = states_to_screen[action_idx]
        self.state = s_utils.string_to_one_hot(new_state_string, self._alphabet)
        ###
        new_state = self.state
        reward = preds[action_idx]
        if new_state_string not in all_measured_seqs:
            self._best_fitness = max(self._best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        self._num_actions += 1
        return uncertainty, new_state_string, reward

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
        landscape: Optional[fasthit.Landscape] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        if self._num_actions == 0:
            # indicates model was reset
            self.initialize_data_structures()
        else:
            # set state to best measured sequence from prior batch
            last_round_num = measured_sequences["round"].max()
            last_batch = measured_sequences[
                measured_sequences["round"] == last_round_num
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
            self.state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
        # generate next batch by picking actions
        self.initial_uncertainty = None
        samples = set()
        prev_cost = self.model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        while self.model.cost - prev_cost < self.model_queries_per_round:
            uncertainty, new_state_string, _ = self.pick_action(all_measured_seqs)
            all_measured_seqs.add(new_state_string)
            samples.add(new_state_string)
            if self.initial_uncertainty is None:
                self.initial_uncertainty = uncertainty
            if uncertainty > 2. * self.initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too
                # uncharted
                sampled_seq = self.Thompson_sample(measured_batch)
                self.state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
                self.initial_uncertainty = None
        if len(samples) < self.expmt_queries_per_round:
            random_sequences = s_utils.generate_random_sequences(
                self._seq_len, self.expmt_queries_per_round - len(samples), self._alphabet
            )
            samples.update(random_sequences)
        # get predicted fitnesses of samples
        samples = list(samples)
        encodings = self.encoder.encode(samples)
        preds = self.model.get_fitness(encodings)
        # train ensemble model before returning samples
        self.train_models()
        return measured_sequences, samples, preds

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

    def _Greedy(self, preds):
        return preds

    def _EI(self, preds):
        """Compute expected improvement."""
        return np.array([max(pred - self._best_fitness, 0) for pred in preds])

    def _UCB(self, preds):
        """Upper confidence bound."""
        discount = 0.01
        return preds - discount * self.model.uncertainties

    def _Thompson(self, preds):
        return np.random.normal(preds, self.model.uncertainties)

    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]


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
        expmt_queries_per_round: int,
        model_queries_per_round: int,
        training_data_size: int,
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        proposal_func: str = "UCB",
        eval_batch_size: int = 256,
    ):
        """Initialize the explorer."""
        name = f"BO_UL_Explorer-proposal_function={proposal_func}"
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
            training_data_size,
            log_file,
        )
        self._alphabet = alphabet
        self._alphabet_len = len(alphabet)
        self._best_fitness = 0
        self._top_sequence = []
        self._seq_len = len(starting_sequence)
        self._eval_batch_size = eval_batch_size
        proposal_funcs = {
            "Greedy": self.propose_sequences_via_greedy,
            "EI": self.propose_sequences_via_ei,
            "Thompson": self.propose_sequences_via_thompson,
            "UCB": self.propose_sequences_via_ucb,
        }
        self._proposal_func = proposal_funcs[proposal_func]

    def reset(self):
        """Reset."""
        self._best_fitness = 0
        self._reset = True

    def propose_sequences_via_greedy(self):
        """Propose a batch of new sequences.
        Based on greedy in the expectation of the Gaussian posterior.
        """
        self.maxima = []
        new_seqs = []
        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            nonlocal new_seqs
            if len(curr_seq) == self._seq_len:
                new_seqs.append(curr_seq)
                if len(new_seqs) >= self._eval_batch_size:
                    encodings = self.encoder.encode(new_seqs)
                    fitness = self.model.get_fitness(encodings)
                    self.maxima.extend(
                        [fitness[i], new_seqs[i]]
                        for i in range(len(new_seqs))
                    )
                    new_seqs = []
            else:
                for char in list(self._alphabet):
                    enum_and_eval(curr_seq + char)
        enum_and_eval("")
        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_ei(self):
        pass

    def propose_sequences_via_thompson(self):
        """Propose a batch of new sequences.
        Based on Thompson sampling with a Gaussian posterior.
        """
        self.maxima = []
        new_seqs = []
        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            nonlocal new_seqs
            if len(curr_seq) == self._seq_len:
                new_seqs.append(curr_seq)
                if len(new_seqs) >= self._eval_batch_size:
                    encodings = self.encoder.encode(new_seqs)
                    fitness = self.model.get_fitness(encodings)
                    estimated_fitness = np.random.normal(fitness, self.model.uncertainties)
                    self.maxima.extend(
                        [estimated_fitness[i], new_seqs[i]]
                        for i in range(len(new_seqs))
                    )
                    new_seqs = []
            else:
                for char in list(self._alphabet):
                    enum_and_eval(curr_seq + char)
        enum_and_eval("")
        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences_via_ucb(self):
        """Propose a batch of new sequences.
        Based on upper confidence bound.
        """
        self.maxima = []
        new_seqs = []
        def enum_and_eval(curr_seq):
            # if we have a full sequence, then let's evaluate
            nonlocal new_seqs
            if len(curr_seq) == self._seq_len:
                new_seqs.append(curr_seq)
                if len(new_seqs) >= self._eval_batch_size:
                    encodings = self.encoder.encode(new_seqs)
                    fitness = self.model.get_fitness(encodings)
                    self.maxima.extend(
                        [fitness[i] + 0.01 * self.model.uncertainties[i], new_seqs[i]]
                        for i in range(len(new_seqs))
                    )
                    new_seqs = []
            else:
                for char in list(self._alphabet):
                    enum_and_eval(curr_seq + char)
        enum_and_eval("")
        # Sort descending based on the value.
        return sorted(self.maxima, reverse=True, key=lambda x: x[0])

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
        landscape: Optional[fasthit.Landscape] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose `batch_size` samples."""
        samples = set()
        new_seqs = self._proposal_func()
        new_states = []
        new_fitnesses = []
        i = 0
        all_measured_seqs = set(measured_sequences["sequence"].values)
        while (len(new_states) < self.expmt_queries_per_round) and i < len(new_seqs):
            new_fitness, new_seq = new_seqs[i]
            if new_seq not in all_measured_seqs:
                new_state = s_utils.string_to_one_hot(new_seq, self._alphabet)
                if new_fitness >= self._best_fitness:
                    self._top_sequence.append((new_fitness, new_state, self.model.cost))
                    self._best_fitness = new_fitness
                samples.add(new_seq)
                all_measured_seqs.add(new_seq)
                new_states.append(new_state)
                new_fitnesses.append(new_fitness)
            i += 1
        return measured_sequences, np.array(list(samples)), np.array(new_fitnesses)

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