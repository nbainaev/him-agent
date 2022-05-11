#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional, Iterator

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sdr_encoders import IntBucketEncoder, IntRandomEncoder
from hima.common.config_utils import extracted_type

from animal_ai_v1_pickle import collect_data
import pickle

from hima.common.utils import clip
from hima.experiments.temporal_pooling.metrics import sdrs_similarity


def v1_output_similarity(output1, output2):
    """
    Corresponding elements of output compared as sets

    @return: mean of similarities
    """
    n = len(output1)
    assert len(output1) == len(output2)
    sim = 0
    for i in range(n):
        sim += sdrs_similarity(output1[i], output2[i])
    return sim/n


class Policy:
    id: int
    _policy: np.ndarray

    def __init__(self, id: int, policy, seed=None):
        self.id = id
        self._policy = policy

    def __iter__(self) -> Iterator[tuple[SparseSdr, SparseSdr]]:
        return iter(self._policy)

    def shuffle(self) -> None:
        ...


class SyntheticGenerator:
    n_states: int
    n_actions: int

    policy_similarity: float
    policy_similarity_std: float
    _rng: Generator

    def __init__(
            self, config: dict,
            n_states: int, n_actions: int,
            state_encoder: str, action_encoder: str,
            policy_similarity: float,
            seed: int,
            policy_similarity_std: float = 0.
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_encoder = resolve_encoder(config, state_encoder, 'state_encoders')
        self.action_encoder = resolve_encoder(config, action_encoder, 'action_encoders')

        self.policy_similarity = policy_similarity
        self.policy_similarity_std = policy_similarity_std
        self._rng = np.random.default_rng(seed)

    def generate_policies(self, n_policies) -> list[Policy]:
        n_states, n_actions = self.n_states, self.n_actions
        rng = self._rng

        base_policy = rng.integers(0, high=n_actions, size=(1, n_states))
        policies = base_policy.repeat(n_policies, axis=0)

        # to-change indices
        for i in range(n_policies - 1):
            if self.policy_similarity_std < 1e-5:
                sim = self.policy_similarity
            else:
                sim = rng.normal(self.policy_similarity, scale=self.policy_similarity_std)
                sim = clip(sim, 0, 1)

            n_states_to_change = int(n_states * (1 - sim))
            if n_states_to_change == 0:
                continue
            indices = rng.choice(n_states, n_states_to_change, replace=False)

            # re-sample actions — from reduced action space (note n_actions-1)
            new_actions = rng.integers(0, n_actions - 1, n_states_to_change)
            old_actions = policies[0][indices]

            # that's how we exclude origin action: |0|1|2| -> |0|.|2|3| — action 1 is excluded
            mask = new_actions >= old_actions
            new_actions[mask] += 1

            # replace origin actions for specified state indices with new actions
            policies[i+1, indices] = new_actions

        states_encoding = [self.state_encoder.encode(s) for s in range(n_states)]
        action_encoding = [self.action_encoder.encode(a) for a in range(n_actions)]

        encoded_policies = []
        for i_policy in range(n_policies):
            policy = []
            for state in range(n_states):
                action = policies[i_policy, state]
                s = states_encoding[state]
                a = action_encoding[action]
                policy.append((s, a))

            encoded_policies.append(Policy(id=i_policy, policy=policy))

        return encoded_policies


class AAIRotationsGenerator:
    def __init__(self, config):
        gen_conf = config['generator']
        dataset_path = gen_conf['dataset_path']
        self.rooms_v1_outputs = pickle.load(open(dataset_path, 'rb'))

    def generate_data(self):
        return self.rooms_v1_outputs

    @staticmethod
    def true_similarities() -> np.ndarray:
        return np.asarray(
            [[1, 0.91, 0.94, 0.94, 0.625],
             [0.91, 1, 0.94, 0.94, 0.68],
             [0.94, 0.94, 1, 0.91, 0.69],
             [0.94, 0.94, 0.91, 1, 0.62],
             [0.625, 0.68, 0.69, 0.62, 1]]
        )

    def v1_similarities(self):
        sim_matrix = np.empty((len(self.rooms_v1_outputs),len(self.rooms_v1_outputs)))
        for i, obs1 in enumerate(self.rooms_v1_outputs):
            for j, obs2 in enumerate(self.rooms_v1_outputs):
                sim_matrix[i][j] = v1_output_similarity(obs1, obs2)
        return sim_matrix


class PolicySelector:
    n_policies: int
    regime: str

    seed: Optional[int]

    def __init__(self, n_policies: int, regime: str, seed: int = None):
        self.n_policies = n_policies
        self.regime = regime
        self.seed = seed

    def __iter__(self):
        if self.regime == 'ordered':
            return range(self.n_policies)
        elif self.regime == 'random':
            assert self.seed is not None, 'seed is expected for random selector'

            rng = np.random.default_rng(self.seed)
            return iter(rng.permutation(self.n_policies))
        else:
            raise KeyError(f'{self.regime} is not supported')


def resolve_data_generator(config: dict):
    seed = config['seed']
    generator_config, generator_type = extracted_type(config['generator'])

    if generator_type == 'synthetic':
        return SyntheticGenerator(config, seed=seed, **generator_config)
    elif generator_type == 'aai_rotation':
        return AAIRotationsGenerator(config)
    else:
        raise KeyError(f'{generator_type} is not supported')


def resolve_encoder(config: dict, key, registry_key: str):
    registry = config[registry_key]
    encoder_config, encoder_type = extracted_type(registry[key])

    if encoder_type == 'int_bucket':
        return IntBucketEncoder(**encoder_config)
    if encoder_type == 'int_random':
        seed = config['seed']
        return IntRandomEncoder(seed=seed, **encoder_config)
    else:
        raise KeyError(f'{encoder_type} is not supported')


def generate_data(n, n_actions, n_states, randomness=1.0, seed=0):
    raw_data = list()
    np.random.seed(seed)
    seed_seq = np.random.randint(0, n_actions, n_states)
    raw_data.append(seed_seq.copy())
    n_replace = int(n_states * randomness)
    for i in range(1, n):
        new_seq = np.random.randint(0, n_actions, n_states)
        if randomness == 1.0:
            raw_data.append(new_seq)
        else:
            indices = np.random.randint(0, n_states, n_replace)
            seed_seq[indices] = new_seq[indices]
            raw_data.append(seed_seq.copy())
    data = [list(zip(range(n_states), x)) for x in raw_data]
    return raw_data, data
