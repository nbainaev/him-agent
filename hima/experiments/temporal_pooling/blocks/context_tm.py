#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


class ContextTemporalMemoryBlock:
    name: str
    feedforward_sds: Sds
    cells_per_column: int
    basal_context_sds: Sds
    apical_feedback_sds: Sds
    cells_sds: Sds

    tm: DelayedFeedbackTM
    tm_config: dict

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
        return active_cells, correctly_predicted_cells
