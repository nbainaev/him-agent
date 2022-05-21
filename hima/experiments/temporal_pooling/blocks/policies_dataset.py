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
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.new.metrics import (
    similarity_matrix,
    standardize_sample_distribution
)


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

    _policies: list[list[set[int]]]

    raw_similarity_matrix_elementwise: np.ndarray
    similarity_matrix_elementwise: np.ndarray
    raw_similarity_elementwise: np.ndarray

    raw_similarity_matrix_union: np.ndarray
    similarity_matrix_union: np.ndarray
    raw_similarity_union: np.ndarray

    raw_similarity_matrix_prefix: np.ndarray
    similarity_matrix_prefix: np.ndarray
    raw_similarity_prefix: np.ndarray

    def __init__(self, policies: list[Policy], actions_sds: Sds):
        self.n_policies = len(policies)
        self._policies = [
            [set(a) for a, s in p]
            for p in policies
        ]
        self.actions_sds = actions_sds

    def compute(self):
        self.raw_similarity_matrix_elementwise = similarity_matrix(
            self._policies, algorithm='elementwise',
            symmetrical=False, sds=self.actions_sds
        )
        self.raw_similarity_elementwise = self.raw_similarity_matrix_elementwise.mean()
        self.similarity_matrix_elementwise = standardize_sample_distribution(
            self.raw_similarity_matrix_elementwise
        )

        self.raw_similarity_matrix_union = similarity_matrix(
            self._policies, algorithm='union', symmetrical=False, sds=self.actions_sds
        )
        self.raw_similarity_union = self.raw_similarity_matrix_union.mean()
        self.similarity_matrix_union = standardize_sample_distribution(
            self.raw_similarity_matrix_union
        )

        self.raw_similarity_matrix_prefix = similarity_matrix(
            self._policies, algorithm='prefix', discount=0.92,
            symmetrical=False, sds=self.actions_sds
        )
        self.raw_similarity_prefix = self.raw_similarity_matrix_prefix.mean()
        self.similarity_matrix_prefix = standardize_sample_distribution(
            self.raw_similarity_matrix_prefix
        )

    @staticmethod
    def step_metrics() -> dict[str, Any]:
        return {}

    def final_metrics(self) -> dict[str, Any]:
        return {
            'raw_sim_mx_el': self.raw_similarity_matrix_elementwise,
            'raw_sim_el': self.raw_similarity_elementwise,
            'sim_mx_el': self.similarity_matrix_elementwise,

            'raw_sim_mx_un': self.raw_similarity_matrix_union,
            'raw_sim_un': self.raw_similarity_union,
            'sim_mx_un': self.similarity_matrix_union,

            'raw_sim_mx_prfx': self.raw_similarity_matrix_prefix,
            'raw_sim_prfx': self.raw_similarity_prefix,
            'sim_mx_prfx': self.similarity_matrix_prefix,
        }


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
            policies: list[Policy], seed: int
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.context_sds = states_sds
        self.output_sds = actions_sds
        self._policies = policies

        self.stats = SyntheticDatasetBlockStats(self._policies, self.output_sds)
        self.stats.compute()
        self._rng = np.random.default_rng(seed)

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[Policy]:
        return iter(self._policies)

    def reset_stats(self):
        ...
