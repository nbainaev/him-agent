#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.blocks.graph import Stream
from hima.experiments.temporal_pooling.new.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.stats.tracker import Tracker, TMetrics


class StreamTracker(Tracker):
    stream: Stream
    trackers: list[Tracker]

    def __init__(
            self, stream: Stream, trackers: list[str], config: StatsMetricsConfig,
            n_sequences: int
    ):
        self.stream = stream
        self.trackers = [
            resolve_tracker(
                tracker_name=tracker_name, sds=stream.sds, n_sequences=n_sequences,
                stats_config=config
            )
            for tracker_name in trackers
        ]

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


def resolve_tracker(
        tracker_name: str, sds: Sds, n_sequences: int, stats_config: StatsMetricsConfig
) -> Tracker:
    if tracker_name == 'sdr':
        from hima.experiments.temporal_pooling.new.stats.sdr_tracker import SdrTracker
        return SdrTracker(sds)
    elif tracker_name == 'cross.offline.el':
        from hima.experiments.temporal_pooling.new.stats.similarity_matrix import \
            OfflineElementwiseSimilarityMatrix
        return OfflineElementwiseSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity
        )
    elif tracker_name == 'cross.offline.prefix':
        from hima.experiments.temporal_pooling.new.stats.similarity_matrix import \
            OfflinePmfSimilarityMatrix
        return OfflinePmfSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            distribution_metrics=stats_config.distribution_metrics,
            pmf_decay=stats_config.pmf_decay
        )
    elif tracker_name == 'cross.online.el':
        from hima.experiments.temporal_pooling.new.stats.similarity_matrix import \
            OnlineElementwiseSimilarityMatrix
        return OnlineElementwiseSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            online_similarity_decay=stats_config.online_similarity_decay
        )
    elif tracker_name == 'cross.online.prefix':
        from hima.experiments.temporal_pooling.new.stats.similarity_matrix import \
            OnlinePmfSimilarityMatrix
        return OnlinePmfSimilarityMatrix(
            n_sequences=n_sequences, sds=sds,
            normalization=stats_config.mae_normalization,
            symmetrical=stats_config.symmetrical_similarity,
            distribution_metrics=stats_config.distribution_metrics,
            online_similarity_decay=stats_config.online_similarity_decay,
            pmf_decay=stats_config.pmf_decay
        )


