from typing import Optional
from namedlist import NO_DEFAULT
import numpy as np
import editdistance as edd
from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo
from ding.envs.common.env_element import EnvElementInfo
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray

from fasthit.utils import sequence_utils as s_utils

@ENV_REGISTRY.register("mutative")
class MutativeEnv(BaseEnv):
    def __init__(self, cfg: dict = {}) -> None:
        ###
        self._alphabet = cfg.alphabet
        self._encoder = cfg.encoder
        self._model = cfg.model
        ###
        self._previous_reward = -float("inf")
        ###
        self._seq = cfg.starting_seq
        encodings = self._encoder.encode([cfg.starting_seq])
        self._state = {
            "sequence": s_utils.string_to_one_hot(self._seq, self._alphabet
            ).astype(np.float32),
            "fitness": self._model.get_fitness(encodings
            ).astype(np.float32),
        }
        self._episode_seqs = set()  # the sequences seen in the current episode
        self._all_seqs = {}
        #self._measured_sequences = {}
        ###
        self._lam = 0.1
        ###
        self._num_step = 0
        self._max_num_steps = cfg.max_num_steps

        #self._final_eval_reward = 0
        #self._count = 0
        ###

    def reset(self, starting_seq: Optional[str] = None):
        if starting_seq is not None:
            self._seq = starting_seq
        self._previous_reward = -float("inf")
        encodings = self._encoder.encode([self._seq])
        self._state = {
            "sequence": s_utils.string_to_one_hot(self._seq, self._alphabet
                ).astype(np.float32),
            "fitness": self._model.get_fitness(encodings
                ).astype(np.float32),
        }
        self._episode_seqs = set()
        self._num_steps = 0
        self._final_eval_reward = 0
        return self._state

    def close(self) -> None:
        pass

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def step(self, action: int) -> BaseEnvTimestep:
        info = {}
        ### Reach maximum env steps
        #"""
        if self._num_steps >= self._max_num_steps:
            #info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(self._state, to_ndarray([0.]), True, info)
        #"""
        ###
        pos = action // len(self._alphabet)
        res = action % len(self._alphabet)
        self._num_steps += 1
        ### No-op
        #"""
        if self._state["sequence"][pos, res] == 1:
            #info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(self._state, to_ndarray([0.]), True, info)
        #"""
        self._state["sequence"][pos] = 0
        self._state["sequence"][pos, res] = 1
        state_string = s_utils.one_hot_to_string(self._state["sequence"], self._alphabet)
        encodings = self._encoder.encode([state_string])
        self._state["fitness"] = self._model.get_fitness(encodings
            ).astype(np.float32)
        self._all_seqs[state_string] = self._state["fitness"].item()
        reward = self._state["fitness"].item() - self._lam * self._sequence_density(
            state_string
        )
        ###
        #"""
        if state_string in self._episode_seqs:
            #self._final_eval_reward += -1
            #info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(self._state, to_ndarray([-1.]), True, info)
        #"""
        self._episode_seqs.add(state_string)
        ###
        #"""
        if reward < self._previous_reward:
            #self._final_eval_reward += reward
            #info['final_eval_reward'] = self._final_eval_reward
            return BaseEnvTimestep(self._state, to_ndarray([reward]), True, info)
        #"""
        self._previous_reward = reward
        ###
        #self._final_eval_reward += reward
        #info['final_eval_reward'] = self._final_eval_reward
        return BaseEnvTimestep(self._state, to_ndarray([reward]), False, info)

    def _sequence_density(self, seq):
        dens = 0
        dist_radius = 2
        for s in self._all_seqs:
            dist = int(edd.eval(s, seq))
            if dist != 0 and dist <= dist_radius:
                dens += self._all_seqs[s] / dist
        return dens

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T(
                (len(self._seq), len(self._alphabet)),
                {
                    'min': 0,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
            act_space=T(
                (1, ),
                {
                    'min': 0,
                    'max': len(self._seq) * len(self._alphabet) - 1,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1, ),
                {
                    'min': 0.0,
                    'max': 1.0
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return "BioSeq MutativeEnv"
