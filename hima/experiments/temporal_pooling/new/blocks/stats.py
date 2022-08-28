#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

from hima.experiments.temporal_pooling.new.blocks.graph import Stream


class StreamStats:
    stream: Stream

    def __init__(self, stream: Stream):
        self.stream = stream

    @property
    def sds(self):
        return self.stream.sds

    def update(self, **kwargs):
        ...

    def step_metrics(self) -> dict[str, Any]:
        return {}

    def aggregate_metrics(self) -> dict[str, Any]:
        ...
