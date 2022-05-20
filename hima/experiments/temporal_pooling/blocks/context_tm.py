#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


class TemporalMemoryBlockStats:
    recall: float

    def __init__(self):
        self.recall = 0.

    def update(self, active_cells: SparseSdr, correctly_predicted_cells: SparseSdr):
        self.recall = safe_divide(correctly_predicted_cells.size, active_cells.size)

    def step_metrics(self) -> dict[str, Any]:
        return {
            'recall': self.recall
        }

    @staticmethod
    def final_metrics() -> dict[str, Any]:
        # TODO: collect repr and distr
        return {}


class ContextTemporalMemoryBlock:
    id: int
    name: str
    feedforward_sds: Sds
    cells_per_column: int
    basal_context_sds: Sds
    apical_feedback_sds: Sds
    cells_sds: Sds

    tm: DelayedFeedbackTM
    tm_config: dict

    stats: TemporalMemoryBlockStats

    _apical_feedback: SparseSdr

    def __init__(self, ff_sds: Sds, bc_sds: Sds, **partially_resolved_tm_config):
        cells_per_column = partially_resolved_tm_config['cells_per_column']

        self.feedforward_sds = ff_sds
        self.cells_per_column = cells_per_column
        self.basal_context_sds = bc_sds
        self.cells_sds = Sds(
            size=self.feedforward_sds.size * cells_per_column,
            active_size=self.feedforward_sds.active_size
        )
        self.tm_config = partially_resolved_tm_config
        self.stats = TemporalMemoryBlockStats()

    @property
    def tag(self) -> str:
        return f'{self.id}_tm'

    @property
    def output_sds(self):
        return self.cells_sds

    def configure_apical_feedback(self, fb_sds, resolved_tm_config):
        self.apical_feedback_sds = fb_sds
        self.tm_config = resolved_tm_config
        self.tm = DelayedFeedbackTM(**self.tm_config)
        self._apical_feedback = []

    def pass_feedback(self, apical_feedback: SparseSdr):
        self._apical_feedback = apical_feedback

    def reset(self):
        self.tm.reset()
        self._apical_feedback = []

    def reset_stats(self):
        self.stats = TemporalMemoryBlockStats()

    def compute(
            self, feedforward_input: SparseSdr, basal_context: SparseSdr, learn: bool
    ) -> tuple[SparseSdr, SparseSdr]:
        tm = self.tm

        tm.set_active_context_cells(basal_context)
        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(self._apical_feedback)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(feedforward_input)
        tm.activate_cells(learn)

        active_cells = np.array(tm.get_active_cells(), copy=True)
        correctly_predicted_cells = np.array(tm.get_correctly_predicted_cells(), copy=True)

        self.stats.update(
            active_cells=active_cells,
            correctly_predicted_cells=correctly_predicted_cells
        )
        return active_cells, correctly_predicted_cells
