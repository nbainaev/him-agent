#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Any

import numpy as np
from htm.bindings.sdr import SDR

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.new.metrics import entropy, representation_from_pmf


class TemporalPoolerBlockStats:
    output_sds: Sds

    current_sdr: set
    aggregate_histogram: np.ndarray
    aggregate_steps: int

    # step metrics
    step_sparsity: float
    step_relative_sparsity: float
    new_cells_relative_ratio: float
    sym_diff_cells_ratio: float

    # aggregate/cluster/sequence metrics
    agg_sparsity: float
    agg_relative_sparsity: float
    active_binary_coverage: float
    active_pmf_coverage: float
    agg_entropy: float
    active_entropy_coverage: float

    def __init__(self, output_sds: Sds):
        self.output_sds = output_sds
        self.current_sdr = set()
        self.aggregate_histogram = np.zeros(self.output_sds.size)
        self.aggregate_steps = 0

    def update(self, current_output_sdr: SparseSdr):
        prev_sdr = self.current_sdr
        curr_sdr = self.current_sdr = set(current_output_sdr)
        curr_sdr_lst = current_output_sdr

        self.aggregate_histogram[curr_sdr_lst] += 1
        self.aggregate_steps += 1

        # step metrics
        curr_sdr_size = len(curr_sdr)
        self.step_sparsity = safe_divide(curr_sdr_size, self.output_sds.size)
        self.step_relative_sparsity = safe_divide(curr_sdr_size, self.output_sds.active_size)
        self.new_cells_relative_ratio = safe_divide(
            len(curr_sdr - prev_sdr), self.output_sds.active_size
        )
        self.sym_diff_cells_ratio = safe_divide(
            len(curr_sdr ^ prev_sdr),
            len(curr_sdr | prev_sdr)
        )

        # aggregate/cluster/sequence metrics
        agg_sdr_size = np.count_nonzero(self.aggregate_histogram)
        agg_pmf = self.aggregate_pmf()

        self.agg_sparsity = safe_divide(agg_sdr_size, self.output_sds.size)
        self.agg_relative_sparsity = safe_divide(agg_sdr_size, self.output_sds.active_size)
        self.active_binary_coverage = safe_divide(curr_sdr_size, agg_sdr_size)
        self.active_pmf_coverage = safe_divide(agg_pmf[curr_sdr_lst].sum(), agg_pmf.sum())

        self.agg_entropy = entropy(agg_pmf, self.output_sds)
        active_entropy = entropy(agg_pmf[curr_sdr_lst], self.output_sds)
        self.active_entropy_coverage = safe_divide(active_entropy, self.agg_entropy)

    def step_metrics(self) -> dict[str, Any]:
        step_metrics = {
            'step/sparsity': self.step_sparsity,
            'step/relative_sparsity': self.step_relative_sparsity,
            'step/new_cells_relative_ratio': self.new_cells_relative_ratio,
            'step/sym_diff_cells_ratio': self.sym_diff_cells_ratio,
        }
        aggregate_metrics = {
            'agg/sparsity': self.agg_sparsity,
            'agg/relative_sparsity': self.agg_relative_sparsity,
            'agg/active_binary_coverage': self.active_binary_coverage,
            'agg/active_pmf_coverage': self.active_pmf_coverage,
            'agg/entropy': self.agg_entropy,
            'agg/active_entropy_coverage': self.active_entropy_coverage,
        }
        return step_metrics | aggregate_metrics

    def final_metrics(self) -> dict[str, Any]:
        distribution = self.aggregate_pmf()

        # noinspection PyTypeChecker
        representative_sdr_lst: list = representation_from_pmf(
            pmf=distribution, sds=self.output_sds
        ).tolist()
        representative_sdr = set(representative_sdr_lst)

        agg_sdr_size = np.count_nonzero(self.aggregate_histogram)
        agg_pmf = self.aggregate_pmf()

        agg_relative_sparsity = safe_divide(agg_sdr_size, self.output_sds.active_size)
        representative_pmf_coverage = agg_pmf[representative_sdr_lst].sum() / agg_pmf.sum()
        return {
            'representative': representative_sdr,
            'distribution': distribution,
            'relative_sparsity': agg_relative_sparsity,
            'representative_pmf_coverage': representative_pmf_coverage
        }

    def aggregate_pmf(self) -> np.ndarray:
        return self.aggregate_histogram / self.aggregate_steps


class TemporalPoolerBlock:
    id: int
    name: str
    feedforward_sds: Sds
    output_sds: Sds

    output_sdr: SparseSdr
    tp: Any
    stats: TemporalPoolerBlockStats

    _input_active_cells: SDR
    _input_predicted_cells: SDR

    def __init__(self, feedforward_sds: Sds, output_sds: Sds, tp: Any):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.tp = tp
        self.stats = TemporalPoolerBlockStats(self.output_sds)

        self.output_sdr = []
        self._input_active_cells = SDR(self.feedforward_sds.size)
        self._input_predicted_cells = SDR(self.feedforward_sds.size)

    @property
    def tag(self) -> str:
        return f'{self.id}_tp'

    def reset(self):
        self.tp.reset()
        self.output_sdr = []

    def reset_stats(self, stats: TemporalPoolerBlockStats = None):
        if stats is None:
            self.stats = TemporalPoolerBlockStats(self.output_sds)
        else:
            self.stats = stats

    def compute(self, active_input: SparseSdr, predicted_input: SparseSdr, learn: bool):
        self._input_active_cells.sparse = active_input.copy()
        self._input_predicted_cells.sparse = predicted_input.copy()

        output_sdr: SDR = self.tp.compute(
            self._input_active_cells, self._input_predicted_cells, learn
        )
        self.output_sdr = np.array(output_sdr.sparse, copy=True)

        self.stats.update(self.output_sdr)
        return self.output_sdr
