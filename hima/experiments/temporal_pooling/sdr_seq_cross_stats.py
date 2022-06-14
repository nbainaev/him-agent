#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any, Optional, Union

import numpy as np

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    standardize_sample_distribution,
    similarity_matrix, sequence_similarity_elementwise, distribution_similarity, DISTR_SIM_PMF,
    DISTR_SIM_KL
)
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats


class SimilarityMatrix:
    tag: str

    raw_mx: np.ndarray
    mx: np.ndarray
    raw_mean: float

    def __init__(self, tag: str, raw_mx: np.ndarray, unbias_func: str):
        self.tag = tag
        self.raw_mx = raw_mx
        self.mx = standardize_sample_distribution(self.raw_mx, unbias_func=unbias_func)
        # self.raw_mean = self.raw_mx.mean()

    def final_metrics(self) -> dict[str, Any]:
        tag = self.tag
        return {
            f'raw_sim_mx_{tag}': self.raw_mx,
            f'sim_mx_{tag}': self.mx,
            # f'avg/raw_sim_{tag}': self.raw_mean,
        }


class OfflineElementwiseSimilarityMatrix(SimilarityMatrix):
    def __init__(
            self, sequences: list[list],
            unbias_func: str, discount: float = None, symmetrical: bool = False
    ):
        n = len(sequences)
        diagonal_mask = np.identity(n, dtype=bool)
        sm = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sm[i, j] = sequence_similarity_elementwise(
                    sequences[i], sequences[j], discount=discount, symmetrical=symmetrical
                )
        raw_mx = np.ma.array(sm, mask=diagonal_mask)

        super(OfflineElementwiseSimilarityMatrix, self).__init__(
            tag='el', raw_mx=raw_mx, unbias_func=unbias_func
        )


class OfflinePmfSimilarityMatrix(SimilarityMatrix):
    def __init__(
            self, pmfs: list[np.ndarray],
            unbias_func: str, algorithm: str, symmetrical: bool = False, sds: Sds = None
    ):
        n = len(pmfs)
        diagonal_mask = np.identity(n, dtype=bool)
        sm = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                sm[i, j] = distribution_similarity(
                    pmfs[i], pmfs[j], algorithm=algorithm, sds=sds, symmetrical=symmetrical
                )
        raw_mx = np.ma.array(sm, mask=diagonal_mask)

        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1_nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        super(OfflinePmfSimilarityMatrix, self).__init__(
            tag=tag, raw_mx=raw_mx, unbias_func=unbias_func
        )


class OnlineSimilarityMatrix:
    tag: str

    sequences_stats: list[Optional[SdrSequenceStats]]

    raw_mx: np.ndarray
    mx: np.ndarray
    raw_mean: float

    def __init__(
            self,
            sequences: Union[list[set[int]], list[np.ndarray], list[list]],
            tag: str, unbias_func: str,
            prefix_algorithm: str, prefix_discount: float,
            symmetrical: bool = False, sds: Sds = None
    ):
        self.tag = tag
        self.raw_mx = similarity_matrix(
            sequences, algorithm=algorithm, symmetrical=symmetrical, sds=sds
        )
        self.mx = standardize_sample_distribution(self.raw_mx, unbias_func=unbias_func)
        self.raw_mean = self.raw_mx.mean()

    def final_metrics(self) -> dict[str, Any]:
        tag = self.tag
        return {
            f'raw/sim_mx_{tag}': self.raw_mx,
            f'avg/raw_sim_{tag}': self.raw_mean,
            f'sim_mx_{tag}': self.mx
        }


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
