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


class TemporalPoolerBlock:
    name: str
    feedforward_sds: Sds
    output_sds: Sds
    tp: Any

    _input_active_cells: SDR
    _input_predicted_cells: SDR

    output_sdr: SparseSdr

    def __init__(self, feedforward_sds: Sds, output_sds: Sds, tp: Any):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.tp = tp

        self._input_active_cells = SDR(self.feedforward_sds.size)
        self._input_predicted_cells = SDR(self.feedforward_sds.size)
        self.output_sdr = []

    def reset(self):
        self.tp.reset()

    def compute(self, active_input: SparseSdr, predicted_input: SparseSdr, learn: bool):
        self._input_active_cells.sparse = active_input.copy()
        self._input_predicted_cells.sparse = predicted_input.copy()

        output_sdr: SDR = self.tp.compute(
            self._input_active_cells, self._input_predicted_cells, learn
        )
        self.output_sdr = np.array(output_sdr.sparse, copy=True)
        return self.output_sdr

