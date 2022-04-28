#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional

import numpy as np

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.utils import safe_divide


# noinspection PyAttributeOutsideInit
class ExperimentStats:
    # current policy id
    policy_id: Optional[int]

    tp_expected_active_size: int
    tp_output_sdr_size: int

    last_representations: dict[int, SparseSdr]
    tp_current_representation: set
    tp_output_distribution: dict[int, DenseSdr]

    def __init__(self, temporal_pooler):
        self.policy_id = None
        self.last_representations = {}
        self.tp_current_representation = set()
        self.tp_output_distribution = {}
        self.tp_output_sdr_size = temporal_pooler.output_sdr_size
        self.tp_expected_active_size = temporal_pooler.n_active_bits

    def on_policy_change(self, policy_id):
        self.policy_id = policy_id
        self.window_size = 1
        self.window_error = 0
        self.whole_active = None
        self.policy_repeat = 0
        self.intra_policy_step = 0
        self.tp_output_distribution.setdefault(
            policy_id, np.empty(self.tp_output_sdr_size, dtype=int)
        ).fill(0)

    def on_policy_repeat(self):
        self.intra_policy_step = 0
        self.policy_repeat += 1

    def on_step(
            self, policy_id: int,
            temporal_memory, temporal_pooler, logger
    ):
        if policy_id != self.policy_id:
            self.on_policy_change(policy_id)

        tm_log = self._get_tm_metrics(temporal_memory)
        tp_log = self._get_tp_metrics(temporal_pooler)
        if logger:
            logger.log(tm_log | tp_log)

        self.intra_policy_step += 1

    # noinspection PyProtectedMember
    def _get_tp_metrics(self, temporal_pooler) -> dict:
        prev_repr = self.tp_current_representation
        curr_repr_lst = temporal_pooler.getUnionSDR().sparse
        curr_repr = set(curr_repr_lst)
        self.tp_current_representation = curr_repr
        # noinspection PyTypeChecker
        self.last_representations[self.policy_id] = curr_repr

        output_distribution = self.tp_output_distribution[self.policy_id]
        output_distribution[curr_repr_lst] += 1

        sparsity = safe_divide(
            len(curr_repr), self.tp_expected_active_size
        )
        new_cells_ratio = safe_divide(
            len(curr_repr - prev_repr), self.tp_expected_active_size
        )
        cells_in_whole = safe_divide(
            len(curr_repr), np.count_nonzero(output_distribution)
        )
        step_difference = safe_divide(
            len(curr_repr ^ prev_repr),
            len(curr_repr | prev_repr)
        )

        return {
            'tp/sparsity': sparsity,
            'tp/new_cells': new_cells_ratio,
            'tp/cells_in_whole': cells_in_whole,
            'tp/step_diff': step_difference
        }

    def _get_tm_metrics(self, temporal_memory) -> dict:
        active_cells: np.ndarray = temporal_memory.get_active_cells()
        predicted_cells: np.ndarray = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.size, active_cells.size)

        return {
            'tm/recall': recall
        }
