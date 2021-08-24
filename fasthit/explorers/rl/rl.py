import numpy as np
import pandas as pd
from functools import partial
from easydict import EasyDict
from typing import Optional, Tuple

from ding.utils import ENV_REGISTRY
from ding.envs.env_manager import BaseEnvManager
from ding.worker.learner import BaseLearner
from .collectors.collector import SampleCollector
from ding.worker.replay_buffer import NaiveReplayBuffer
from ding.torch_utils.data_helper import to_ndarray
from ding.utils import deep_merge_dicts

import fasthit
from fasthit.utils import sequence_utils as s_utils

from .ppo import PPOPolicy, PPOOffPolicy
from .models.policy_models import VAC

class RL(fasthit.Explorer):
    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int, # default: 384
        model_queries_per_round: int, # default: 3200
        starting_sequence: str,
        cfg: dict,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
    ):
        name = f"RL-DyNAPPO"

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
        self._cfg = EasyDict(cfg)
        self._alphabet = alphabet
        self._seq_len = len(self.starting_sequence)
        self._num_model_rounds = 1

        from .envs import MutativeEnv
        env_fn = ENV_REGISTRY.get("mutative")

        env_cfg = self._cfg.env
        env_cfg.update(dict(
            alphabet=self._alphabet,
            starting_seq=self._starting_sequence,
            encoder=self.encoder,
            model=self.model,
            max_num_steps=self.expmt_queries_per_round,
        ))

        env_manager_cfg = deep_merge_dicts(BaseEnvManager.default_config(), env_cfg.manager)
        collector_env_num = env_cfg.collector_env_num
        collector_env = BaseEnvManager(
            [partial(
                env_fn,
                cfg=env_cfg,
            ) for _ in range(collector_env_num)],
            env_manager_cfg
        )

        policy_cls = PPOPolicy if self._cfg.policy.on_policy else PPOOffPolicy
        policy_cfg = deep_merge_dicts(policy_cls.default_config(), self._cfg.policy)
        model = VAC(**policy_cfg.model)
        policy = policy_cls(policy_cfg, model=model)
        
        learner_cfg = deep_merge_dicts(BaseLearner.default_config(), policy_cfg.learn.learner)
        self._learner = BaseLearner(
            learner_cfg,
            policy.learn_mode
        )

        collector_cfg = deep_merge_dicts(SampleCollector.default_config(), policy_cfg.collect.collector)
        self._collector = SampleCollector(
            collector_cfg,
            self._alphabet,
            env=collector_env,
            policy=policy.collect_mode,
        )

        buffer_cfg = deep_merge_dicts(NaiveReplayBuffer.default_config(), policy_cfg.other.replay_buffer)
        self._replay_buffer = NaiveReplayBuffer(
            buffer_cfg,
        )

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        sequences = {}
        previous_model_cost = self.model.cost
        while (
            self.model.cost - previous_model_cost < self.model_queries_per_round
        ):
            new_data = self._collector.collect(train_iter=self._learner.train_iter)
            self._replay_buffer.push(new_data, cur_collector_envstep=self._collector.envstep)
            sequences.update({
                s_utils.one_hot_to_string(
                    to_ndarray(new["next_obs"]["sequence"].cpu()),
                    self._alphabet
                ) : new["next_obs"]["fitness"].item()
                for new in new_data
            })
        
        for i in range(self._cfg.policy.learn.update_per_collect):
            # Learner will train ``update_per_collect`` times in one iteration.
            train_data = self._replay_buffer.sample(
                self._learner.policy.get_attribute('batch_size'), self._learner.train_iter
            )
            self._learner.train(train_data, self._collector.envstep)
            if self._learner.policy.get_attribute('priority'):
                self._replay_buffer.update(self._learner.priority_info)
        if self._cfg.policy.on_policy:
            self._replay_buffer.clear()
        ###
        sequences = {
            seq: fitness
            for seq, fitness in sequences.items()
            if seq not in set(measured_sequences["sequence"])
        }
        new_seqs = np.array(list(sequences.keys()))
        preds = np.array(list(sequences.values()))
        sorted_order = np.argsort(preds)[: -self.expmt_queries_per_round - 1 : -1]
        return measured_sequences, new_seqs[sorted_order], preds[sorted_order]

    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences
