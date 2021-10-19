import numpy as np
import pandas as pd
from functools import partial
from easydict import EasyDict
from typing import Optional, Tuple

from ding.config import compile_config
from ding.envs.env.base_env import get_env_cls
from ding.envs import create_env_manager
from ding.worker.learner import create_learner
from ding.worker.collector import create_serial_collector
from ding.worker.replay_buffer import create_buffer
from ding.worker.coordinator import BaseSerialCommander
from ding.policy import create_policy
from ding.model import create_model
from ding.torch_utils.data_helper import to_ndarray

import fasthit
from fasthit.utils import sequence_utils as s_utils


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
        self._alphabet = alphabet
        ###
        cfg = EasyDict(cfg)
        create_cfg = cfg.pop("create_cfg")
        create_cfg.policy.type = create_cfg.policy.type + '_command'
        self._cfg = compile_config(
            cfg,
            seed=42,
            auto=True,
            create_cfg=create_cfg
        )
        ###
        name = f"RL-{self._cfg.policy.type}"
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
        ###
        env_cfg = self._cfg.env
        env_fn = get_env_cls(env_cfg)
        env_cfg.update(dict(
            alphabet=self._alphabet,
            starting_seq=self._starting_sequence,
            encoder=self.encoder,
            model=self.model,
            max_num_steps=self.model_queries_per_round,
        ))
        ###
        collector_env_num = env_cfg.collector_env_num
        env_manager_cfg = env_cfg.manager
        collector_env = create_env_manager(
            env_manager_cfg,
            [partial(
                env_fn,
                cfg=env_cfg,
            ) for _ in range(collector_env_num)],
        )
        ###
        policy_cfg = self._cfg.policy
        model = create_model(policy_cfg.model)
        policy = create_policy(
            policy_cfg,
            model=model,
            enable_field=["learn", "collect", "command"],
        )
        ###
        self._learner = create_learner(
            policy_cfg.learn.learner,
            policy=policy.learn_mode,
            exp_name=self._cfg.exp_name,
        )
        ###
        self._collector = create_serial_collector(
            policy_cfg.collect.collector,
            alphabet=self._alphabet,
            env=collector_env,
            policy=policy.collect_mode,
            exp_name=self._cfg.exp_name,
        )
        ###
        self._replay_buffer = create_buffer(
            policy_cfg.other.replay_buffer,
            exp_name=self._cfg.exp_name,
        )

        self._commander = BaseSerialCommander(
            policy_cfg.other.commander,
            self._learner,
            self._collector,
            None,
            self._replay_buffer,
            policy=policy.command_mode,
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
            collect_kwargs = self._commander.step()
            new_data = self._collector.collect(
                self._cfg.policy.collect.n_sample,
                train_iter=self._learner.train_iter,
                policy_kwargs=collect_kwargs
            )
            self._replay_buffer.push(new_data, cur_collector_envstep=self._collector.envstep)
            sequences.update({
                s_utils.one_hot_to_string(
                    to_ndarray(new["obs"]["sequence"].cpu()),
                    self._alphabet
                ) : new["obs"]["fitness"].item()
                for new in new_data
            })
        ###
        for _ in range(self._cfg.policy.learn.update_per_collect):
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
