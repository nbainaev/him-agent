#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator, Any

import numpy as np
from numpy.random import Generator

from hima.common.config_utils import check_all_resolved
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import clip
from hima.experiments.temporal_pooling.new.blocks.computational_graph import Block
from hima.experiments.temporal_pooling.new.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling.new.blocks.dataset_resolver import resolve_encoder
from hima.experiments.temporal_pooling.new.sdr_seq_cross_stats import OfflineElementwiseSimilarityMatrix
from hima.experiments.temporal_pooling.new.stats_config import StatsMetricsConfig


class Sequence:
    """Sequence of SDRs."""
    id: int
    _sequence: list[SparseSdr]

    def __init__(self, id_: int, sequence):
        self.id = id_
        self._sequence = sequence

    def __iter__(self) -> Iterator[dict]:
        for sdr in self._sequence:
            yield dict(feedforward=sdr)

    def __len__(self):
        return len(self._sequence)


class SyntheticSequencesDatasetBlockStats(BlockStats):
    n_sequences: int
    actions_sds: Sds
    sequences: list[list[set[int]]]
    cross_stats: OfflineElementwiseSimilarityMatrix

    def __init__(self, sequences: list[Sequence], actions_sds: Sds, stats_config: StatsMetricsConfig):
        super(SyntheticSequencesDatasetBlockStats, self).__init__(output_sds=actions_sds)

        self.n_sequences = len(sequences)
        self.sequences = [
            [set(sdr) for sdr in seq]
            for seq in sequences
        ]
        self.actions_sds = actions_sds
        self.cross_stats = OfflineElementwiseSimilarityMatrix(
            sequences=self.sequences,
            normalization=stats_config.normalization,
            discount=stats_config.prefix_similarity_discount,
            symmetrical=False
        )

    def final_metrics(self) -> dict[str, Any]:
        return self.cross_stats.final_metrics()


class SyntheticSequencesDatasetBlock(Block):
    family = "generator"

    n_values: int
    stats: SyntheticSequencesDatasetBlockStats

    _sequences: list[Sequence]

    def __init__(self, id_: int, name: str, n_values: int, sequences: list[Sequence]):
        super(SyntheticSequencesDatasetBlock, self).__init__(id_, name)

        self.n_values = n_values
        self._sequences = sequences

    def build(self, stats_config: StatsMetricsConfig = None):
        assert check_all_resolved(self.output_sds)

        if stats_config is not None:
            self.stats = SyntheticSequencesDatasetBlockStats(
                self._sequences, self.output_sds, stats_config
            )

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self._sequences)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self.sdr['output'] = data['feedforward']

    def reset_stats(self):
        ...


class SyntheticSequencesGenerator:
    family = "generator"

    sequence_length: int
    n_values: int
    values_sds: Sds

    sequence_similarity: float
    sequence_similarity_std: float

    seed: int
    _rng: Generator

    def __init__(
            self, config: dict,
            sequence_length: int, n_values: int, active_size: int,
            value_encoder: str,
            sequence_similarity: float,
            seed: int,
            sequence_similarity_std: float = 0.
    ):
        self.sequence_length = sequence_length
        self.n_values = n_values
        self.value_encoder = resolve_encoder(
            config, value_encoder, 'encoder',
            n_values=self.n_values,
            active_size=active_size,
            seed=seed
        )
        self.values_sds = self.value_encoder.output_sds

        self.sequence_similarity = sequence_similarity
        self.sequence_similarity_std = sequence_similarity_std

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def generate_sequences(self, n_sequences: int) -> list[Sequence]:
        values_encoding = [self.value_encoder.encode(x) for x in range(self.n_values)]

        sequences = generate_synthetic_sequences(
            n_sequences=n_sequences,
            sequence_length=self.sequence_length,
            n_values=self.n_values,
            seed=self.seed,
            sequence_similarity=self.sequence_similarity,
            sequence_similarity_std=self.sequence_similarity_std,
        )
        encoded_sequences = [
            Sequence(
                id_=i_sequence,
                sequence=[values_encoding[x] for x in sequence]
            )
            for i_sequence, sequence in enumerate(sequences)
        ]

        return encoded_sequences

    def build_block(
            self, block_id: int, block_name: str,
            sequences: list[Sequence], stats_config: StatsMetricsConfig
    ) -> SyntheticSequencesDatasetBlock:
        block = SyntheticSequencesDatasetBlock(
            id_=block_id, name=block_name,
            n_values=self.n_values, sequences=sequences
        )
        block.resolve_sds('output', self.values_sds)
        block.build(stats_config=stats_config)
        return block


def generate_synthetic_sequences(
        n_sequences: int, sequence_length: int, n_values: int, seed: int,
        sequence_similarity: float,
        sequence_similarity_std: float = 0.
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    base_sequence = rng.integers(0, high=n_values, size=(1, sequence_length))
    sequences = base_sequence.repeat(n_sequences, axis=0)

    # to-change indices
    for i in range(n_sequences - 1):
        if sequence_similarity_std < 1e-5:
            sim = sequence_similarity
        else:
            sim = rng.normal(sequence_similarity, scale=sequence_similarity_std)
            sim = clip(sim, 0, 1)

        n_values_to_change = int(sequence_length * (1 - sim))
        if n_values_to_change == 0:
            continue
        indices = rng.choice(sequence_length, n_values_to_change, replace=False)

        # re-sample values from reduced value space (note n_values-1 below)
        new_values = rng.integers(0, n_values - 1, n_values_to_change)
        old_values = sequences[0][indices]

        # that's how we exclude origin value: |0|1|2| -> |0|.|2|3| — value 1 is excluded
        mask = new_values >= old_values
        new_values[mask] += 1

        # replace origin values for specified positions with new values
        sequences[i+1, indices] = new_values

    return sequences
