#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any, Optional

import numpy as np

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    standardize_sample_distribution,
    similarity_matrix
)
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats


class SimilarityMatrix:
    raw_mx: np.ndarray
    mx: np.ndarray
    raw_mean: float

    def __init__(self, raw_mx: np.ndarray, unbias_func: str):
        self.raw_mx = raw_mx
        self.mx = standardize_sample_distribution(self.raw_mx, unbias_func=unbias_func)
        self.raw_mean = self.raw_mx.mean()


class SdrSequencesOfflineCrossStats:
    """Computes similarity matrices in offline fashion, as if all sequences are known ATM."""
    sds: Sds

    sim_mx_elementwise: SimilarityMatrix
    sim_mx_union: SimilarityMatrix
    sim_mx_prefix: SimilarityMatrix

    def __init__(
            self, sequences: list[list], sds: Sds,
            prefix_algorithm: str, prefix_discount: float, unbias_func: str
    ):
        self.sds = sds

        raw_sim_mx_elementwise = similarity_matrix(
            sequences, algorithm='elementwise', symmetrical=False, sds=sds
        )
        raw_sim_mx_union = similarity_matrix(
            sequences, algorithm='union.point_similarity', symmetrical=False, sds=sds
        )
        raw_sim_mx_prefix = similarity_matrix(
            sequences, algorithm=prefix_algorithm, discount=prefix_discount,
            symmetrical=False, sds=sds
        )

        self.sim_mx_elementwise = SimilarityMatrix(raw_sim_mx_elementwise, unbias_func=unbias_func)
        self.sim_mx_union = SimilarityMatrix(raw_sim_mx_union, unbias_func=unbias_func)
        self.sim_mx_prefix = SimilarityMatrix(raw_sim_mx_prefix, unbias_func=unbias_func)

    def final_metrics(self) -> dict[str, Any]:
        return {
            'raw/sim_mx_el': self.sim_mx_elementwise.raw_mx,
            'raw/sim_mx_un': self.sim_mx_union.raw_mx,
            'raw/sim_mx_prfx': self.sim_mx_prefix.raw_mx,

            'avg/raw_sim_el': self.sim_mx_elementwise.raw_mean,
            'avg/raw_sim_un': self.sim_mx_union.raw_mean,
            'avg/raw_sim_prfx': self.sim_mx_prefix.raw_mean,

            'sim_mx_el': self.sim_mx_elementwise.mx,
            'sim_mx_un': self.sim_mx_union.mx,
            'sim_mx_prfx': self.sim_mx_prefix.mx,
        }


class SdrSequencesOnlineCrossStats:
    """Computes similarity matrices in online fashion. Only currently available info is used."""
    sds: Sds

    sequences_stats: list[Optional[SdrSequenceStats]]

    sim_mx_elementwise: SimilarityMatrix
    sim_mx_union: SimilarityMatrix
    sim_mx_prefix: SimilarityMatrix

    raw_sim_mx_elementwise: np.ndarray
    raw_sim_mx_union: np.ndarray
    raw_sim_mx_prefix: np.ndarray

    prefix_algorithm: str
    prefix_discount: float
    unbias_func: str

    prefix_algorithm: str
    prefix_discount: float
    unbias_func: str

    def __init__(
            self, sds: Sds, n_sequences: int,
            prefix_algorithm: str, prefix_discount: float, unbias_func: str
    ):
        self.sds = sds
        self.sequences_stats = [None] * n_sequences
        self.prefix_algorithm = prefix_algorithm
        self.prefix_discount = prefix_discount
        self.unbias_func = unbias_func

    def step(self):
        ...

    def final_metrics(self) -> dict[str, Any]:
        return {
            'raw_sim_mx_el': self.sim_mx_elementwise.raw_mx,
            'raw_sim_el': self.sim_mx_elementwise.raw_mean,
            'sim_mx_el': self.sim_mx_elementwise.mx,

            'raw_sim_mx_un': self.sim_mx_union.raw_mx,
            'raw_sim_un': self.sim_mx_union.raw_mean,
            'sim_mx_un': self.sim_mx_union.mx,

            'raw_sim_mx_prfx': self.sim_mx_prefix.raw_mx,
            'raw_sim_prfx': self.sim_mx_prefix.raw_mean,
            'sim_mx_prfx': self.sim_mx_prefix.mx,
        }
