#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
from typing import Iterator, Any

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.new.metrics import (
    similarity_matrix,
    standardize_sample_distribution
)
from hima.experiments.temporal_pooling.sds_stats import SdsStats


class RoomObservationSequence:
    """Sequence of observations."""
    id: int

    _observations: list[SparseSdr]

    def __init__(self, id_: int, observations):
        self.id = id_
        self._observations = observations

    def __iter__(self) -> Iterator[SparseSdr]:
        return iter(self._observations)

    def __len__(self):
        return len(self._observations)


class AnimalAiDatasetBlockStats:
    n_sequences: int
    output_sds: Sds

    _observation_sequences: list[list[set[int]]]

    sds_stats: SdsStats

    # epoch final metrics
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
        self.sds_stats = SdsStats(self.output_sds)

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
            self._observation_sequences, algorithm='union.point_similarity',
            symmetrical=False, sds=self.output_sds
        )
        self.raw_similarity_union = self.raw_similarity_matrix_union.mean()
        self.similarity_matrix_union = standardize_sample_distribution(
            self.raw_similarity_matrix_union
        )

        self.raw_similarity_matrix_prefix = similarity_matrix(
            self._observation_sequences, algorithm='prefix.point_similarity', discount=0.92,
            symmetrical=False, sds=self.output_sds
        )
        self.raw_similarity_prefix = self.raw_similarity_matrix_prefix.mean()
        self.similarity_matrix_prefix = standardize_sample_distribution(
            self.raw_similarity_matrix_prefix
        )

    def reset(self):
        self.sds_stats = SdsStats(self.output_sds)

    def update(self, current_output_sdr: SparseSdr):
        self.sds_stats.update(current_output_sdr)

    def step_metrics(self) -> dict[str, Any]:
        return self.sds_stats.step_metrics()

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
    n_observations_per_sequence: int
    output_sds: Sds

    stats: AnimalAiDatasetBlockStats

    _observation_sequences: list[RoomObservationSequence]

    def __init__(self, sds: Sds, observation_sequences: list[RoomObservationSequence]):
        self.output_sds = sds
        self._observation_sequences = observation_sequences
        self.n_sequences = len(self._observation_sequences)
        self.n_observations_per_sequence = len(self._observation_sequences[0])

        self.stats = AnimalAiDatasetBlockStats(self._observation_sequences, self.output_sds)
        self.stats.compute()

    @property
    def tag(self) -> str:
        return f'{self.id}_in'

    def __iter__(self) -> Iterator[RoomObservationSequence]:
        return iter(self._observation_sequences)

    def update_stats(self, observation: SparseSdr):
        self.stats.update(observation)

    def reset_stats(self):
        self.stats.reset()


class AAIRotationsGenerator:
    TDataset = list[list[SDR]]

    n_sequences: int
    n_rotations: int
    v1_output_sequences: TDataset
    v1_output_sds: Sds

    def __init__(
            self, sds: Sds.TShortNotation, filepath: str
    ):
        self.v1_output_sequences = self.restore_dataset(filepath)
        self.v1_output_sds = Sds(short_notation=sds)

        self.n_sequences = len(self.v1_output_sequences)
        self.n_rotations = len(self.v1_output_sequences[0])

    def generate_data(self, n_sequences: int = None, n_observations_per_sequence: int = None):
        n_sequences = isnone(n_sequences, self.n_sequences)
        n_rotations = isnone(n_observations_per_sequence, self.n_rotations)

        # NB: take only requested N of sequences and rotations per sequence
        observation_sequences = [
            RoomObservationSequence(
                id_=ind,
                observations=[
                    np.array(obs.sparse, copy=True)
                    for obs in observations[:n_rotations]
                ]
            )
            for ind, observations in enumerate(self.v1_output_sequences[:n_sequences])
        ]
        return AnimalAiDatasetBlock(
            sds=self.v1_output_sds,
            observation_sequences=observation_sequences,
        )

    @staticmethod
    def restore_dataset(filepath: str) -> TDataset:
        with open(filepath, 'rb') as dataset_io:
            return pickle.load(dataset_io)
