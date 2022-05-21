#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import pickle
from typing import Optional, Iterator

import numpy as np
from htm.bindings.sdr import SDR
from numpy.random import Generator

from hima.common.sds import Sds
from hima.common.utils import clip
from hima.experiments.temporal_pooling.blocks.aai_dataset import (
    RoomObservationSequence,
    AnimalAiDatasetBlock
)
from hima.experiments.temporal_pooling.blocks.policies_dataset import SyntheticDatasetBlock, Policy
from hima.experiments.temporal_pooling.config_resolvers import resolve_encoder


class SyntheticGenerator:
    n_states: int
    n_actions: int

    policy_similarity: float
    policy_similarity_std: float

    seed: int
    _rng: Generator

    def __init__(
            self, config: dict,
            n_states: int, n_actions: int, active_size: int,
            state_encoder: str, action_encoder: str,
            policy_similarity: float,
            seed: int,
            policy_similarity_std: float = 0.
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_encoder = resolve_encoder(
            config, state_encoder, 'encoders',
            n_values=self.n_states,
            active_size=active_size,
            seed=seed
        )
        self.states_sds = self.state_encoder.output_sds

        self.action_encoder = resolve_encoder(
            config, action_encoder, 'encoders',
            n_values=self.n_actions,
            active_size=active_size,
            seed=seed
        )
        self.actions_sds = self.action_encoder.output_sds

        self.policy_similarity = policy_similarity
        self.policy_similarity_std = policy_similarity_std

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_policies(self, n_policies) -> SyntheticDatasetBlock:
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
                policy.append((a, s))

            encoded_policies.append(Policy(id_=i_policy, policy=policy))

        return SyntheticDatasetBlock(
            n_states=self.n_states, states_sds=self.states_sds,
            n_actions=self.n_actions, actions_sds=self.actions_sds,
            policies=encoded_policies, seed=self.seed
        )


class AAIRotationsGenerator:
    TDataset = list[list[SDR]]

    v1_output_sequences: TDataset
    v1_output_sds: Sds

    def __init__(self, sds: Sds.TShortNotation, filepath: str):
        self.v1_output_sequences = self.restore_dataset(filepath)
        self.v1_output_sds = Sds(short_notation=sds)

    def generate_data(self):
        observation_sequences = [
            RoomObservationSequence(
                id_=ind,
                observations=[np.array(obs.sparse, copy=True) for obs in observations]
            )
            for ind, observations in enumerate(self.v1_output_sequences)
        ]
        return AnimalAiDatasetBlock(
            sds=self.v1_output_sds,
            observation_sequences=observation_sequences,
        )

    @staticmethod
    def restore_dataset(filepath: str) -> TDataset:
        with open(filepath, 'rb') as dataset_io:
            return pickle.load(dataset_io)


class SequenceSelector:
    n_elements: int
    regime: str
    seed: Optional[int]

    indices: np.ndarray

    def __init__(self, n_elements: int, regime: str, seed: int = None):
        self.n_elements = n_elements
        self.regime = regime
        self.seed = seed
        self.indices = self._select_order()

    def reshuffle(self):
        self.seed = np.random.default_rng(self.seed).integers(100000)
        self.indices = self._select_order()

    def __iter__(self) -> Iterator[int]:
        if self.regime == 'unordered':
            # re-shuffle each time making the order to be random
            self.reshuffle()

        return iter(self.indices)

    def _select_order(self) -> np.ndarray:
        if self.regime == 'ordered':
            return np.arange(self.n_elements)
        elif self.regime == 'shuffled' or self.regime == 'unordered':
            rng = np.random.default_rng(self.seed)
            return rng.permutation(self.n_elements)
        else:
            raise KeyError(f'{self.regime} is not supported')


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
