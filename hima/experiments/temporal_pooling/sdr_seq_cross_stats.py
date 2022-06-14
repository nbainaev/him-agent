#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any, Optional

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    standardize_sample_distribution,
    sequence_similarity_elementwise, distribution_similarity, DISTR_SIM_PMF,
    DISTR_SIM_KL
)
from hima.experiments.temporal_pooling.sdr_seq_stats import SdrSequenceStats


class SimilarityMatrix:
    tag: str

    _raw_mx: np.ma.MaskedArray
    _mx: Optional[np.ndarray]

    unbias_func: str
    online: bool

    def __init__(self, tag: str, raw_mx: np.ma.MaskedArray, unbias_func: str, online: bool = False):
        self.tag = tag
        self.online = online
        self.unbias_func = unbias_func
        self._raw_mx = raw_mx
        self._mx = None

    def final_metrics(self) -> dict[str, Any]:
        tag = self.tag
        if self.online or self._mx is None:
            self._mx = standardize_sample_distribution(self._raw_mx, unbias_func=self.unbias_func)

        return {
            f'raw_sim_mx_{tag}': self._raw_mx,
            f'sim_mx_{tag}': self._mx,
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


class OnlineElementwiseSimilarityMatrix(SimilarityMatrix):
    symmetrical: Optional[bool]
    discount: Optional[float]

    sequences_stats: list[Optional[SdrSequenceStats]]
    cum_sim: np.ndarray
    counts: np.ndarray

    current_i_seq: Optional[int]
    current_stats: Optional[SdrSequenceStats]

    def __init__(
            self, n_sequences: int, unbias_func: str, discount: float, symmetrical: bool = False
    ):
        self.symmetrical = symmetrical
        self.discount = discount

        self.sequences_stats = [None] * n_sequences
        self.current_i_seq = -1
        self.current_stats = None

        raw_mx = np.empty((n_sequences, n_sequences))
        full_mask = np.ones_like(raw_mx, dtype=bool)

        self.cum_sim = np.zeros_like(raw_mx)
        self.counts = np.zeros_like(raw_mx, dtype=int)

        raw_mx = np.ma.array(raw_mx, mask=full_mask)
        super(OnlineElementwiseSimilarityMatrix, self).__init__(
            tag='prfx_el', raw_mx=raw_mx, unbias_func=unbias_func, online=True
        )

    def new_sequence(self, sequence_id: int, stats: SdrSequenceStats):
        if self.current_stats is not None:
            self.sequences_stats[self.current_i_seq] = self.current_stats

        self.current_stats = stats
        self.current_i_seq = sequence_id

    def update(self):
        # NB: current stats has already been updated
        n = len(self.sequences_stats)
        x = self.current_stats.sdr_history
        for j in range(n):
            if self.sequences_stats[j] is None:
                continue

            y = self.sequences_stats[j].sdr_history[:len(x)]
            sim = sequence_similarity_elementwise(
                x, y, discount=self.discount, symmetrical=self.symmetrical
            )
            self._update(j, sim)

    def _update(self, j, sim):
        i = self.current_i_seq

        self.cum_sim[i, j] += sim
        self.counts[i, j] += 1
        self._raw_mx[i, j] = self.cum_sim[i, j] / self.counts[i, j]
        self._raw_mx.mask[i, j] = False


class OnlinePmfSimilarityMatrix(SimilarityMatrix):
    SdrSequence = Optional[list[SparseSdr]]

    symmetrical: Optional[bool]
    discount: Optional[float]

    sequences: list[SdrSequence]
    cum_sim: np.ndarray
    counts: np.ndarray

    current_i_seq: Optional[int]
    current_seq: SdrSequence

    def __init__(
            self, n_sequences: int, unbias_func: str, discount: float, symmetrical: bool = False
    ):
        self.symmetrical = symmetrical
        self.discount = discount

        self.sequences = [None] * n_sequences
        self.current_i_seq = -1
        self.current_seq = None

        raw_mx = np.empty((n_sequences, n_sequences))
        full_mask = np.ones_like(raw_mx, dtype=bool)

        self.cum_sim = np.zeros_like(raw_mx)
        self.counts = np.zeros_like(raw_mx, dtype=int)

        raw_mx = np.ma.array(raw_mx, mask=full_mask)
        super(OnlinePmfSimilarityMatrix, self).__init__(
            tag='prfx_el', raw_mx=raw_mx, unbias_func=unbias_func, online=True
        )

    def new_sequence(self, sequence_id: int):
        if self.current_seq is not None:
            self.sequences[self.current_i_seq] = self.current_seq

        self.current_seq = []
        self.current_i_seq = sequence_id

    def update(self, sdr: SparseSdr):
        self.current_seq.append(set(sdr))

        # update metrics
        n_seqs = len(self.sequences)
        prefix_size = len(self.current_seq)

        x = self.current_seq
        for j in range(n_seqs):
            if self.sequences[j] is None:
                continue

            y = self.sequences[j][:prefix_size]
            sim = sequence_similarity_elementwise(
                x, y, discount=self.discount, symmetrical=self.symmetrical
            )
            self._update(j, sim)

    def _update(self, j, sim):
        i = self.current_i_seq

        self.cum_sim[i, j] += sim
        self.counts[i, j] += 1
        self._raw_mx[i, j] = self.cum_sim[i, j] / self.counts[i, j]
        self._raw_mx.mask[i, j] = False


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
            'raw_sim_mx_el': self.sim_mx_elementwise._raw_mx,
            'raw_sim_el': self.sim_mx_elementwise.raw_mean,
            'sim_mx_el': self.sim_mx_elementwise._mx,

            'raw_sim_mx_un': self.sim_mx_union._raw_mx,
            'raw_sim_un': self.sim_mx_union.raw_mean,
            'sim_mx_un': self.sim_mx_union._mx,

            'raw_sim_mx_prfx': self.sim_mx_prefix._raw_mx,
            'raw_sim_prfx': self.sim_mx_prefix.raw_mean,
            'sim_mx_prfx': self.sim_mx_prefix._mx,
        }
