#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Iterator

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import clip
from hima.experiments.temporal_pooling.new.resolvers.encoder import resolve_encoder
from hima.experiments.temporal_pooling.new.blocks.graph import Block


class Sequence:
    """Sequence of SDRs."""
    id: int
    raw_sequence: list[SparseSdr]

    def __init__(self, id_: int, sequence):
        self.id = id_
        self.raw_sequence = sequence

    def __iter__(self) -> Iterator[dict]:
        for sdr in self.raw_sequence:
            yield dict(input=sdr)

    def __len__(self):
        return len(self.raw_sequence)


class SyntheticSequencesDatasetBlock(Block):
    OUTPUT = 'output'

    family = "generator"
    supported_streams = {OUTPUT}

    n_values: int
    _sequences: list[Sequence]

    def __init__(
            self, id_: int, name: str, n_values: int, sequences: list[Sequence],
            values_sds: Sds
    ):
        super(SyntheticSequencesDatasetBlock, self).__init__(id_, name)

        self.n_values = n_values
        self._sequences = sequences
        self.register_stream(self.OUTPUT).resolve_sds(values_sds)

    def reset(self, **kwargs):
        super(SyntheticSequencesDatasetBlock, self).reset(**kwargs)

    def build(self):
        pass

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self._sequences)

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        ...

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
            config, value_encoder,
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

    def make_block(
            self, block_id: int, block_name: str, sequences: list[Sequence]
    ) -> SyntheticSequencesDatasetBlock:
        return SyntheticSequencesDatasetBlock(
            id_=block_id, name=block_name,
            n_values=self.n_values,
            sequences=sequences,
            values_sds=self.values_sds
        )


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
