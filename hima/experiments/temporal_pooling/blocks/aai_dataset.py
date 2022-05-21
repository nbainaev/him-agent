#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator, Any

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    similarity_matrix,
    standardize_sample_distribution
)


class RoomObservationSequence:
    """Sequence of observations."""
    id: int
    _observations: list[SparseSdr]

    def __init__(self, id_: int, observations):
        self.id = id_
        self._observations = observations

    def __iter__(self) -> Iterator[SparseSdr]:
        return iter(self._observations)


class AnimalAiDatasetBlockStats:
    n_sequences: int
    output_sds: Sds

    _observation_sequences: list[list[set[int]]]

    raw_similarity_matrix_elementwise: np.ndarray
    similarity_matrix_elementwise: np.ndarray
    raw_similarity_elementwise: np.ndarray

    raw_similarity_matrix_union: np.ndarray
    similarity_matrix_union: np.ndarray
    raw_similarity_union: np.ndarray

    raw_similarity_matrix_prefix: np.ndarray
    similarity_matrix_prefix: np.ndarray
    raw_similarity_prefix: np.ndarray

    def __init__(self, observation_sequences: list[RoomObservationSequence], sds: Sds):
        self.n_sequences = len(observation_sequences)
        self._observation_sequences = [
            [set(obs) for obs in seq]
            for seq in observation_sequences
        ]
        self.output_sds = sds

    @staticmethod
    def distance_based_similarities() -> np.ndarray:
        return np.asarray(
            [[1, 0.91, 0.94, 0.94, 0.625],
             [0.91, 1, 0.94, 0.94, 0.68],
             [0.94, 0.94, 1, 0.91, 0.69],
             [0.94, 0.94, 0.91, 1, 0.62],
             [0.625, 0.68, 0.69, 0.62, 1]]
        )

    def compute(self):
        self.raw_similarity_matrix_elementwise = similarity_matrix(
            self._observation_sequences, algorithm='elementwise',
            symmetrical=False, sds=self.output_sds
        )
        self.raw_similarity_elementwise = self.raw_similarity_matrix_elementwise.mean()
        self.similarity_matrix_elementwise = standardize_sample_distribution(
            self.raw_similarity_matrix_elementwise
        )

        self.raw_similarity_matrix_union = similarity_matrix(
            self._observation_sequences, algorithm='union', symmetrical=False, sds=self.output_sds
        )
        self.raw_similarity_union = self.raw_similarity_matrix_union.mean()
        self.similarity_matrix_union = standardize_sample_distribution(
            self.raw_similarity_matrix_union
        )

        self.raw_similarity_matrix_prefix = similarity_matrix(
            self._observation_sequences, algorithm='prefix', discount=0.92,
            symmetrical=False, sds=self.output_sds
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


class AnimalAiDatasetBlock:
    id: int
    name: str

    n_sequences: int
    output_sds: Sds

    stats: AnimalAiDatasetBlockStats

    _observation_sequences: list[RoomObservationSequence]

    def __init__(self, sds: Sds, observation_sequences: list[RoomObservationSequence]):
        self.output_sds = sds
        self._observation_sequences = observation_sequences
        self.n_sequences = len(self._observation_sequences)

        self.stats = AnimalAiDatasetBlockStats(self._observation_sequences, self.output_sds)
        self.stats.compute()

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[RoomObservationSequence]:
        return iter(self._observation_sequences)

    def reset_stats(self):
        ...
