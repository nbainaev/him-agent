#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class Block:
    kind: str = "base_block"
    id: int
    name: str

    feedforward_sds: Sds
    output_sds: Sds

    output_sdr: SparseSdr

    def __init__(self, feedforward_sds: Sds, output_sds: Sds):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.output_sdr = []

    @property
    def tag(self):
        return f'{self.id}_{self.kind}'

