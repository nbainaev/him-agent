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

    def new_sequence(self, sequence_id: int):
        ...

    def update(self, **kwargs):
        ...

    def final_metrics(self) -> dict[str, Any]:
        tag = self.tag
        if self._mx is None:
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
    SdrSequence = Optional[list[SparseSdr]]

    symmetrical: Optional[bool]
    discount: Optional[float]

    sequences: list[SdrSequence]
    cum_sim: np.ndarray
    step: float

    current_i_seq: Optional[int]
    current_seq: SdrSequence

    def __init__(
            self, n_sequences: int, unbias_func: str, discount: float, symmetrical: bool
    ):
        self.symmetrical = symmetrical
        self.discount = discount

        self.sequences = [None] * n_sequences
        self.current_i_seq = -1
        self.current_seq = None

        self.cum_sim = np.zeros(n_sequences)
        self.step = 0

        raw_mx = np.empty((n_sequences, n_sequences))
        mask = np.ones_like(raw_mx, dtype=bool)
        raw_mx = np.ma.array(raw_mx, mask=mask)
        raw_mx.soften_mask()

        super(OnlineElementwiseSimilarityMatrix, self).__init__(
            tag='prfx_el', raw_mx=raw_mx, unbias_func=unbias_func, online=True
        )

    def new_sequence(self, sequence_id: int):
        if self.current_seq is not None:
            self.sequences[self.current_i_seq] = self.current_seq

        self.cum_sim.fill(0)
        self.step = 0
        self.current_seq = []
        self.current_i_seq = sequence_id

    def update(self, sdr: SparseSdr):
        self.current_seq.append(set(sdr))
        self.step += 1

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
        self.cum_sim[j] += sim
        self._raw_mx[i, j] = self.cum_sim[j] / self.step

    def final_metrics(self) -> dict[str, Any]:
        self._mx = standardize_sample_distribution(self._raw_mx, unbias_func=self.unbias_func)
        return super(OnlineElementwiseSimilarityMatrix, self).final_metrics()


class OnlinePmfSimilarityMatrix(SimilarityMatrix):
    SeqHistogram = Optional[np.ndarray]

    sds: Sds
    symmetrical: Optional[bool]
    discount: Optional[float]
    algorithm: str

    histograms: list[SeqHistogram]
    cum_sim: np.ndarray
    step: float

    current_i_seq: Optional[int]
    current_seq_hist: SeqHistogram

    def __init__(
            self, n_sequences: int, sds: Sds,
            unbias_func: str, discount: float, symmetrical: bool, algorithm: str
    ):
        self.sds = sds
        self.symmetrical = symmetrical
        self.discount = discount

        self.histograms = [None] * n_sequences
        self.current_i_seq = -1
        self.current_seq_hist = None

        self.cum_sim = np.zeros(n_sequences)
        self.step = 0

        self.algorithm = algorithm
        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1_nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        raw_mx = np.empty((n_sequences, n_sequences))
        mask = np.ones_like(raw_mx, dtype=bool)
        raw_mx = np.ma.array(raw_mx, mask=mask, hard_mask=False)

        super(OnlinePmfSimilarityMatrix, self).__init__(
            tag=f'prfx_{tag}', raw_mx=raw_mx, unbias_func=unbias_func, online=True
        )

    def new_sequence(self, sequence_id: int):
        if self.current_seq_hist is not None:
            self.histograms[self.current_i_seq] = self.current_seq_hist

        self.cum_sim.fill(0)
        self.step = 0
        self.current_seq_hist = np.zeros(self.sds.size)
        self.current_i_seq = sequence_id

    def update(self, sdr: SparseSdr):
        # discount preserving `hist / step = 1 * active_size`
        self.current_seq_hist *= self.discount
        self.step *= self.discount
        # add new sdr to pmf
        self.current_seq_hist[list(sdr)] += 1
        self.step += 1

        n = len(self.histograms)
        x = self.current_seq_hist

        for j in range(n):
            if self.histograms[j] is None:
                # print(f'- {j}')
                continue

            y = self.histograms[j]
            sim = distribution_similarity(
                x, y, algorithm=self.algorithm, sds=self.sds, symmetrical=self.symmetrical
            )
            self._update(j, sim)

    def _update(self, j, sim):
        i = self.current_i_seq
        self.cum_sim[j] += sim
        self._raw_mx[i, j] = self.cum_sim[j] / self.step

    def final_metrics(self) -> dict[str, Any]:
        self._mx = standardize_sample_distribution(self._raw_mx, unbias_func=self.unbias_func)
        return super(OnlinePmfSimilarityMatrix, self).final_metrics()
