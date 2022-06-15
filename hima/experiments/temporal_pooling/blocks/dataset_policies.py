#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator, Any

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import clip
from hima.experiments.temporal_pooling.blocks.dataset_resolver import resolve_encoder
from hima.experiments.temporal_pooling.sdr_seq_cross_stats import OfflineElementwiseSimilarityMatrix
from hima.experiments.temporal_pooling.stats_config import StatsMetricsConfig


class Policy:
    """Sequence of (action, state)."""
    id: int
    _policy: list[tuple[SparseSdr, SparseSdr]]

    def __init__(self, id_: int, policy):
        self.id = id_
        self._policy = policy

    def __iter__(self) -> Iterator[tuple[SparseSdr, SparseSdr]]:
        return iter(self._policy)

    def __len__(self):
        return len(self._policy)


class SyntheticDatasetBlockStats:
    n_policies: int
    actions_sds: Sds
    policies: list[list[set[int]]]
    cross_stats: OfflineElementwiseSimilarityMatrix

    def __init__(self, policies: list[Policy], actions_sds: Sds, stats_config: StatsMetricsConfig):
        self.n_policies = len(policies)
        self.policies = [
            [set(a) for a, s in p]
            for p in policies
        ]
        self.actions_sds = actions_sds
        self.cross_stats = OfflineElementwiseSimilarityMatrix(
            sequences=self.policies,
            unbias_func=stats_config.normalization_unbias,
            discount=stats_config.prefix_similarity_discount,
            symmetrical=False
        )

    @staticmethod
    def step_metrics() -> dict[str, Any]:
        return {}

    def final_metrics(self) -> dict[str, Any]:
        return self.cross_stats.final_metrics()


class SyntheticDatasetBlock:
    id: int
    name: str
    n_states: int
    n_actions: int

    context_sds: Sds
    output_sds: Sds

    stats: SyntheticDatasetBlockStats

    _policies: list[Policy]
    _rng: Generator

    def __init__(
            self, n_states: int, states_sds: Sds, n_actions: int, actions_sds: Sds,
            policies: list[Policy], seed: int, stats_config: StatsMetricsConfig
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.context_sds = states_sds
        self.output_sds = actions_sds
        self._policies = policies

        self.stats = SyntheticDatasetBlockStats(self._policies, self.output_sds, stats_config)
        self._rng = np.random.default_rng(seed)

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies)

    def reset_stats(self):
        ...


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

    def generate_policies(
            self, n_policies, stats_config: StatsMetricsConfig
    ) -> SyntheticDatasetBlock:
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
            policies=encoded_policies, seed=self.seed, stats_config=stats_config
        )
