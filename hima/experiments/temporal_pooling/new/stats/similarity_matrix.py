#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Union

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    standardize_sample_distribution,
    sequence_similarity_elementwise, distribution_similarity, DISTR_SIM_PMF,
    DISTR_SIM_KL, NO_NORMALIZATION, aggregate_pmf
)
from hima.experiments.temporal_pooling.new.stats.sdr_tracker import SdrSequence, SeqHistogram
from hima.experiments.temporal_pooling.new.stats.tracker import Tracker, TMetrics


class SimilarityMatrix(Tracker):
    tag: str

    n_sequences: int
    sds: Sds
    mx: Union[np.ndarray, np.ma.MaskedArray]

    # mean: mean/std; min: min/max-min; no: no normalization
    normalization: str

    # 1. offline means we first collect full sequence then find its cross similarity to the others
    # 2. online means we calculate cross similarity at each step (i.e. for each of the sequence's
    #   prefix) and then average it over time steps to get the result
    # NB: online version estimates how similar the current sequence looks like to the already
    #   seen sequences in real time, while the offline version makes posterior comparison.
    online: bool

    symmetrical: bool
    discount: float

    current_seq_id: Optional[int]
    current_seq: SdrSequence

    def __init__(
            self, tag: str, n_sequences: int, sds: Sds,
            normalization: str = NO_NORMALIZATION,
            discount: float = 1., symmetrical: bool = False
    ):
        self.tag = tag
        self.n_sequences = n_sequences
        self.sds = sds
        self.normalization = normalization

        self.symmetrical = symmetrical
        self.discount = discount

        self.current_seq_id = None
        self.current_seq = []

        # FIXME: do we need diag?
        raw_mx = np.empty((n_sequences, n_sequences))
        mask = np.ones_like(raw_mx, dtype=bool)
        self.mx = np.ma.array(raw_mx, mask=mask)

    def on_sequence_started(self, sequence_id: int):
        self.current_seq = []
        self.current_seq_id = sequence_id

    def on_epoch_started(self):
        # NB: consider configure whether propagate the accumulated stats from the previous epoch.
        #   For not propagating re-init the object
        pass

    def on_step(self, sdr: SparseSdr):
        # NB: both online and offline similarity calculation goes on seq finish
        # because similarity matrix isn't published at each time step
        self.current_seq.append(sdr)

    def on_sequence_finished(self):
        # it's expected to update similarity matrix now
        pass

    def step_metrics(self) -> TMetrics:
        return {}

    def aggregate_metrics(self) -> TMetrics:
        tag = self.tag
        agg_metrics = {
            f'raw_sim_mx_{tag}': self.mx,
        }

        if self.normalization != NO_NORMALIZATION:
            agg_metrics[f'sim_mx_{tag}'] = standardize_sample_distribution(
                self.mx, normalization=self.normalization
            )

        return agg_metrics


class OfflineElementwiseSimilarityMatrix(SimilarityMatrix):
    sequences: list[Optional[SdrSequence]]

    def __init__(
            self, n_sequences: int, sds: Sds,
            normalization: str, discount: float, symmetrical: bool
    ):
        self.sequences = [None] * n_sequences
        super(OfflineElementwiseSimilarityMatrix, self).__init__(
            tag='el', n_sequences=n_sequences, sds=sds,
            normalization=normalization, discount=discount, symmetrical=symmetrical
        )

    def on_sequence_finished(self):
        self.sequences[self.current_seq_id] = self.current_seq

    def on_epoch_finished(self):
        n = self.n_sequences
        diagonal_mask = np.identity(n, dtype=bool)
        self.mx = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                self.mx[i, j] = sequence_similarity_elementwise(
                    s1=self.sequences[i],
                    s2=self.sequences[j],
                    discount=self.discount, symmetrical=self.symmetrical
                )
        self.mx = np.ma.array(self.mx, mask=diagonal_mask)


class OfflinePmfSimilarityMatrix(SimilarityMatrix):
    algorithm: str
    pmfs: list[Optional[SeqHistogram]]

    def __init__(
            self, n_sequences: int, sds: Sds,
            normalization: str, discount: float, symmetrical: bool, algorithm: str
    ):
        self.pmfs = [None] * n_sequences

        self.algorithm = algorithm
        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1-nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        super(OfflinePmfSimilarityMatrix, self).__init__(
            tag=tag, n_sequences=n_sequences, sds=sds,
            normalization=normalization, discount=discount, symmetrical=symmetrical
        )

    def on_sequence_finished(self):
        self.pmfs[self.current_seq_id] = aggregate_pmf(seq=self.current_seq, sds=self.sds)

    def on_epoch_finished(self):
        n = self.n_sequences
        diagonal_mask = np.identity(n, dtype=bool)
        mx = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                mx[i, j] = distribution_similarity(
                    p=self.pmfs[i],
                    q=self.pmfs[j],
                    algorithm=self.algorithm, sds=self.sds, symmetrical=self.symmetrical
                )
        self.mx = np.ma.array(mx, mask=diagonal_mask)


class OnlineElementwiseSimilarityMatrix(SimilarityMatrix):
    sequences: list[Optional[SdrSequence]]

    def __init__(
            self, n_sequences: int, sds: Sds,
            normalization: str, discount: float, symmetrical: bool
    ):
        self.sequences = [None] * n_sequences
        super(OnlineElementwiseSimilarityMatrix, self).__init__(
            tag='prfx_el', n_sequences=n_sequences, sds=sds,
            normalization=normalization, discount=discount, symmetrical=symmetrical
        )

    def on_sequence_finished(self):
        n_seq_elements = len(self.current_seq)

        self.current_seq = [set(sdr) for sdr in self.current_seq]
        similarity_sum = np.zeros(self.n_sequences)

        for step in range(n_seq_elements):
            prefix_size = step + 1
            for j in range(self.n_sequences):
                if self.sequences[j] is None:
                    continue

                similarity_sum[j] += sequence_similarity_elementwise(
                    s1=self.current_seq[:prefix_size],
                    s2=self.sequences[j][:prefix_size],
                    discount=self.discount, symmetrical=self.symmetrical
                )

        # store mean similarity
        mean_similarity = similarity_sum / n_seq_elements
        for j in range(self.n_sequences):
            if self.sequences[j] is None:
                continue
            self.mx[self.current_seq_id][j] = mean_similarity[j]

        # store sequence; MUST BE stored after similarity calculations to prevent rewriting prev seq
        self.sequences[self.current_seq_id] = self.current_seq


class OnlinePmfSimilarityMatrix(SimilarityMatrix):
    algorithm: str
    pmfs: list[Optional[SeqHistogram]]

    def __init__(
            self, n_sequences: int, sds: Sds,
            normalization: str, discount: float, symmetrical: bool, algorithm: str
    ):
        self.pmfs = [None] * n_sequences

        self.algorithm = algorithm
        if algorithm == DISTR_SIM_PMF:
            tag = 'pmf'
        elif algorithm == DISTR_SIM_KL:
            tag = '1-nkl'
        else:
            raise KeyError(f'Algorithm {algorithm} is not supported')

        super(OnlinePmfSimilarityMatrix, self).__init__(
            tag=tag, n_sequences=n_sequences, sds=sds,
            normalization=normalization, discount=discount, symmetrical=symmetrical
        )

    def on_sequence_finished(self):
        n_seq_elements = len(self.current_seq)

        similarity_sum = np.zeros(self.n_sequences)
        prefix_histogram = np.zeros(self.sds.size)
        prefix_histogram_sum = 0

        # calculate similarity for every prefix pmf against all stored pmfs
        prefix_pmf = 1
        for sdr in self.current_seq:
            prefix_histogram_sum = self._update_pmf(
                sdr, histogram=prefix_histogram, histogram_sum=prefix_histogram_sum
            )
            prefix_pmf = prefix_histogram / prefix_histogram_sum

            for j in range(self.n_sequences):
                if self.pmfs[j] is None:
                    continue
                similarity_sum[j] += distribution_similarity(
                    p=prefix_pmf, q=self.pmfs[j],
                    algorithm=self.algorithm, sds=self.sds, symmetrical=self.symmetrical
                )

        # update mean similarity for the entire `current_seq_id` row
        mean_similarity = similarity_sum / n_seq_elements
        for j in range(self.n_sequences):
            if self.pmfs[j] is None:
                continue
            self.mx[self.current_seq_id][j] = mean_similarity[j]

        # store pmf; MUST BE stored after similarity calculations to prevent rewriting prev pmf
        # because in real time we can only compare to the already seen sequences
        self.pmfs[self.current_seq_id] = prefix_pmf

    def _update_pmf(self, sdr: SparseSdr, histogram: SeqHistogram, histogram_sum: float):
        if self.discount < 1.:
            # discount preserving `hist / step = 1 * active_size`
            histogram *= self.discount
            histogram_sum *= self.discount

        # add new sdr to pmf
        histogram[sdr] += 1
        return histogram_sum + 1
