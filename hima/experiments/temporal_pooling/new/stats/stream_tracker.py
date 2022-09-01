#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config_utils import TConfig
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.new.blocks.graph import Stream
from hima.experiments.temporal_pooling.new.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.stats.tracker import Tracker, TMetrics


class StreamTracker(Tracker):
    stream: Stream

    def __init__(self, stream: Stream, track_streams: TConfig, config: StatsMetricsConfig):
        self.stream = stream

    @property
    def sds(self):
        return self.stream.sds

    def on_sequence_started(self, sequence_id: int):
        pass

    def on_epoch_started(self):
        pass

    def on_step(self, sdr: SparseSdr):
        pass

    def on_sequence_finished(self):
        pass

    def on_epoch_finished(self):
        pass

    def step_metrics(self) -> TMetrics:
        pass

    def aggregate_metrics(self) -> TMetrics:
        pass
