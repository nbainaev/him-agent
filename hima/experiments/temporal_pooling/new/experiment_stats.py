#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional, Any

import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.common.utils import ensure_list
from hima.experiments.temporal_pooling.new.blocks.graph import Block, Stream
from hima.experiments.temporal_pooling.new.metrics import (
    multiplicative_loss, DISTR_SIM_PMF
)
from hima.experiments.temporal_pooling.new.sdr_seq_cross_stats import (
    OnlinePmfSimilarityMatrix, OfflinePmfSimilarityMatrix
)
from hima.experiments.temporal_pooling.new.stats.stream_tracker import StreamTracker
from hima.experiments.temporal_pooling.new.stats.config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.utils import rename_dict_keys


class RunProgress:
    epoch: int
    step: int

    def __init__(self):
        self.epoch = -1
        self.step = -1

    def next_epoch(self):
        self.epoch += 1

    def next_step(self):
        self.step += 1


class ExperimentStats:
    TSequenceId = int
    TStreamName = str
    TMetricsName = str

    n_sequences: int
    progress: RunProgress
    logger: Optional[Run]
    stats_config: StatsMetricsConfig

    stream_trackers: dict[TStreamName, StreamTracker]
    current_sequence_id: int

    debug: bool
    logging_temporally_disabled: bool

    def __init__(
            self, n_sequences: int, progress: RunProgress, logger: Optional[Run],
            blocks: dict[str, Block], track_streams: TConfig,
            stats_config: StatsMetricsConfig, debug: bool
    ):
        self.n_sequences = n_sequences
        self.progress = progress
        self.logger = logger
        self.stats_config = stats_config
        self.debug = debug
        self.logging_temporally_disabled = True
        self.current_sequence_id = -1

        self.stream_trackers = self._make_stream_trackers(
            track_streams=track_streams, blocks=blocks,
            stats_config=stats_config, n_sequences=n_sequences
        )

    @staticmethod
    def _make_stream_trackers(
            track_streams: TConfig, blocks: dict[str, Block],
            stats_config: StatsMetricsConfig, n_sequences: int
    ) -> dict[TStreamName, StreamTracker]:
        trackers = {}
        for stream_name in track_streams:
            stream = parse_stream_name(stream_name, blocks)
            stream_trackers_list = track_streams[stream_name]
            tracker = StreamTracker(
                stream=stream, trackers=stream_trackers_list,
                config=stats_config, n_sequences=n_sequences
            )
            trackers[tracker.name] = tracker
        return trackers

    def define_metrics(self):
        if not self.logger:
            return

        self.logger.define_metric('epoch')
        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            self.logger.define_metric(f'{tracker.name}/epoch/*', step_metric='epoch')

    def on_epoch_started(self):
        for name in self.stream_trackers:
            self.stream_trackers[name].on_epoch_started()

    def on_sequence_started(self, sequence_id: int, logging_scheduled: bool):
        self.logging_temporally_disabled = not logging_scheduled
        if sequence_id == self.current_sequence_id:
            return

        self.current_sequence_id = sequence_id
        for name in self.stream_trackers:
            self.stream_trackers[name].on_sequence_started(sequence_id)

    def on_sequence_finished(self):
        for name in self.stream_trackers:
            self.stream_trackers[name].on_sequence_finished()

    def on_step(self):
        if self.logger is None and not self.debug:
            return
        if self.logging_temporally_disabled:
            return

        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            tracker.on_step(tracker.stream.sdr)

        metrics = {
            'epoch': self.progress.epoch
        }
        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            metrics |= rename_dict_keys(tracker.step_metrics(), add_prefix=f'{tracker.name}/')

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def on_epoch_finished(self, logging_scheduled: bool):
        if not self.logger and not self.debug:
            return
        if not logging_scheduled:
            return

        for name in self.stream_trackers:
            self.stream_trackers[name].on_epoch_finished()

        metrics = {}
        diff_metrics = []
        optimized_metrics = []
        for name in self.stream_trackers:
            tracker = self.stream_trackers[name]
            block = tracker.stream.block

            if block.family == 'generator':
                self.summarize_input(tracker, metrics, diff_metrics)
            elif block.family == 'spatial_pooler':
                self.summarize_sp(block, metrics)
            elif block.family == 'temporal_memory':
                ...
            elif block.family == 'temporal_pooler':
                self.summarize_tp(block, metrics, diff_metrics, optimized_metrics)
            else:
                raise KeyError(f'Block {block.family} is not supported')

        # metrics |= self.summarize_similarity_errors(diff_metrics, optimized_metrics)

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def summarize_input(self, stream_stats: StreamTracker, metrics: dict, diff_metrics: list):
        stream = stream_stats.stream
        stream_metrics = stream_stats.aggregate_metrics()

        # block_diff_metrics = {
        #     'raw_sim_mx': stream_metrics['raw_sim_mx_el'],
        #     'sim_mx': stream_metrics['sim_mx_el']
        # }
        # diff_metrics.append((stream.name, block_diff_metrics))

        stream_metrics = rename_dict_keys(stream_metrics, add_prefix=f'{stream.fullname}/epoch/')
        self.transform_sim_mx_to_plots(stream_metrics)

        metrics |= stream_metrics

    def summarize_sp(self, block, metrics: dict):
        block_metrics = self._collect_block_final_stats(block)

        # noinspection PyUnresolvedReferences
        pmfs = [
            self.stats_registry[seq_id][block.stream_tag(stream)].seq_stats.aggregate_pmf()
            for seq_id in range(self.n_sequences)
            for stream in block.stats
        ]
        # offline pmf similarity matrices sim_mx
        offline_pmf_similarity = OfflinePmfSimilarityMatrix(
            pmfs, sds=block.output_sds,
            normalization=self.stats_config.normalization,
            algorithm=DISTR_SIM_PMF, symmetrical=self.stats_config.symmetrical_similarity
        )
        block_metrics |= offline_pmf_similarity.aggregate_metrics()

        # online pmf similarity matrices
        for block_online_similarity_matrix in self.cross_stats_registry[block.name].values():
            block_metrics |= block_online_similarity_matrix.aggregate_metrics()

        block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
        self.transform_sim_mx_to_plots(block_metrics)

        metrics |= block_metrics

    def summarize_tp(self, block, metrics: dict, diff_metrics: list, optimized_metrics: list):
        block_metrics = self._collect_block_final_stats(block)
        optimized_metrics.append(block_metrics['mean_pmf_coverage'])

        # noinspection PyUnresolvedReferences
        pmfs = [
            self.stats_registry[seq_id][block.stream_tag(stream)].seq_stats.aggregate_pmf()
            for seq_id in range(self.n_sequences)
            for stream in block.stats
        ]
        # offline pmf similarity matrices sim_mx
        offline_pmf_similarity_matrix = OfflinePmfSimilarityMatrix(
            pmfs, sds=block.output_sds,
            normalization=self.stats_config.normalization,
            algorithm=DISTR_SIM_PMF, symmetrical=self.stats_config.symmetrical_similarity
        )
        offline_pmf_similarity = offline_pmf_similarity_matrix.aggregate_metrics()

        block_metrics |= offline_pmf_similarity
        block_diff_metrics = {
            'raw_sim_mx': offline_pmf_similarity['raw_sim_mx_pmf'],
            'sim_mx': offline_pmf_similarity['sim_mx_pmf']
        }
        diff_metrics.append((block.tag, block_diff_metrics))

        # online pmf similarity matrices
        for block_online_similarity_matrix in self.cross_stats_registry[block.name].values():
            block_metrics |= block_online_similarity_matrix.aggregate_metrics()

        # track first TP pmf coverage
        if block.tag[0] in {'2', 3}:
            pmf_coverage = block_metrics['mean_pmf_coverage']
            key = f'{block.tag}_mean_pmf_coverage'
            metrics[key] = pmf_coverage

        block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
        self.transform_sim_mx_to_plots(block_metrics)
        metrics |= block_metrics

    def summarize_similarity_errors(self, diff_metrics, optimized_metrics):
        input_tag, input_sims = diff_metrics[0]
        metrics = {
            sim_key: {input_tag: input_sims[sim_key]}
            for sim_key in input_sims
        }

        loss_components = []
        i = 0
        for block_tag, block_sim_metrics in diff_metrics[1:]:
            for metric_key in block_sim_metrics:
                sim_mx = block_sim_metrics[metric_key]

                metrics[metric_key][block_tag] = sim_mx
                abs_err_mx = np.ma.abs(sim_mx - input_sims[metric_key])
                metrics[metric_key][f'{block_tag}_abs_err'] = abs_err_mx

                mae = abs_err_mx.mean()
                if metric_key.startswith('raw_'):
                    metrics[f'{block_tag}/epoch/similarity_mae'] = mae
                else:
                    metrics[f'{block_tag}/epoch/similarity_smae'] = mae

            if block_tag.endswith('_tp'):
                key = 'mae' if self.stats_config.loss_on_mae else 'smae'
                mae = metrics[f'{block_tag}/epoch/similarity_{key}']

                # duplicate target metric to the general wandb panel
                metrics[f'{block_tag}_similarity_{key}'] = mae

                # store to compute loss later
                pmf_coverage = optimized_metrics[i]
                loss_components.append((mae, pmf_coverage))
                i += 1

        result = {}
        for metric_key in metrics.keys():
            metric = metrics[metric_key]
            if isinstance(metric, dict):
                # dict of similarity matrices
                result[f'diff/{metric_key}'] = plot_heatmaps_row(**metric)
            else:
                result[metric_key] = metric

        result['loss'] = compute_loss(loss_components, self.stats_config.loss_layer_discount)
        return result

    def _collect_block_final_stats(self, block) -> dict[str, Any]:
        result = {}
        # collect/reorder from (seq_id, block, metric) -> (block, metric, seq_id)
        for seq_id in range(self.n_sequences):
            for stream in block.stats:
                block_stat = self.stats_registry[seq_id][block.stream_tag(stream)]
                final_metrics = block_stat.final_metrics()
                for metric_key in final_metrics:
                    if metric_key not in result:
                        result[metric_key] = [None]*self.n_sequences
                    result[metric_key][seq_id] = final_metrics[metric_key]

        for metric_key in result:
            if isinstance(result[metric_key][0], np.ndarray):
                result[metric_key] = np.vstack(result[metric_key])
            else:
                result[metric_key] = np.mean(result[metric_key])
        return result

    def notify_cross_stats_new_sequence(self, sequence_id: int):
        for name in self.tracked_streams:
            stream = self.tracked_streams[name]
            stream_cross_stats = self.cross_stats_registry[stream.fullname]

            for online_similarity_matrix in stream_cross_stats.values():
                online_similarity_matrix.on_new_sequence(sequence_id=sequence_id)

    def _init_cross_stats(self):
        self.cross_stats_registry = {}

        for name in self.tracked_streams:
            stream = self.tracked_streams[name]
            block = stream.block

            if block.family == 'generator':
                self.cross_stats_registry[stream.fullname] = {
                    'online_pmf': OnlinePmfSimilarityMatrix(
                        n_sequences=self.n_sequences,
                        sds=stream.sds,
                        normalization=self.stats_config.normalization,
                        discount=self.stats_config.prefix_similarity_discount,
                        symmetrical=self.stats_config.symmetrical_similarity,
                        algorithm=DISTR_SIM_PMF
                    ),
                }
            elif block.family == 'spatial_pooler':
                self.cross_stats_registry[block.name] = {}
                self.cross_stats_registry[block.name] = {
                    # 'online_el': OnlineElementwiseSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity
                    # ),
                    'online_pmf': OnlinePmfSimilarityMatrix(
                        n_sequences=self.n_sequences,
                        sds=stream.sds,
                        normalization=self.stats_config.normalization,
                        discount=self.stats_config.prefix_similarity_discount,
                        symmetrical=self.stats_config.symmetrical_similarity,
                        algorithm=DISTR_SIM_PMF
                    ),
                    # FIXME: kl-div shows peculiar similarity matrix, investigate later
                    # 'online_kl': OnlinePmfSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     sds=block.output_sds,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity,
                    #     algorithm=DISTR_SIM_KL
                    # )
                }
            elif block.family.startswith('temporal_memory'):
                self.cross_stats_registry[block.name] = {}
            elif block.family.startswith('temporal_pooler'):
                self.cross_stats_registry[block.name] = {
                    # 'online_el': OnlineElementwiseSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity
                    # ),
                    'online_pmf': OnlinePmfSimilarityMatrix(
                        n_sequences=self.n_sequences,
                        sds=block.output_sds,
                        normalization=self.stats_config.normalization,
                        discount=self.stats_config.prefix_similarity_discount,
                        symmetrical=self.stats_config.symmetrical_similarity,
                        algorithm=DISTR_SIM_PMF
                    ),
                    # FIXME: kl-div shows peculiar similarity matrix, investigate later
                    # 'online_kl': OnlinePmfSimilarityMatrix(
                    #     n_sequences=self.n_sequences,
                    #     sds=block.output_sds,
                    #     unbias_func=self.stats_config.normalization_unbias,
                    #     discount=self.stats_config.prefix_similarity_discount,
                    #     symmetrical=self.stats_config.symmetrical_similarity,
                    #     algorithm=DISTR_SIM_KL
                    # )
                }
            else:
                raise KeyError(f'Block {block.family} is not supported')

    @staticmethod
    def transform_sim_mx_to_plots(metrics):
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, np.ndarray) and metric_value.ndim == 2:
                metrics[metric_key] = plot_single_heatmap(metric_value)


def parse_stream_name(stream_name: str, blocks: dict[str, Block]) -> Optional[Stream]:
    block_name, stream_name = stream_name.split('.')
    if block_name not in blocks:
        # skip unused blocks
        return None
    return blocks[block_name].streams[stream_name]


def compute_loss(components, layer_discount) -> float:
    gamma = 1
    loss = 0
    for mae, pmf_coverage in components:
        loss += gamma * multiplicative_loss(mae, pmf_coverage)
        gamma *= layer_discount

    return loss


HEATMAP_SIDE_SIZE = 7


def plot_single_heatmap(repr_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(HEATMAP_SIDE_SIZE+1, HEATMAP_SIDE_SIZE-1))
    plot_heatmap(repr_matrix, ax)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmaps_row(**sim_matrices):
    n = len(sim_matrices)
    fig, axes = plt.subplots(
        nrows=1, ncols=n, sharey='all',
        figsize=(HEATMAP_SIDE_SIZE * n + 1, HEATMAP_SIDE_SIZE - 1)
    )

    axes = ensure_list(axes)
    for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
        plot_heatmap(sim_matrix, ax)
        ax.set_title(name, size=10)

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def plot_heatmap(heatmap: np.ndarray, ax):
    v_min, v_max = calculate_heatmap_value_boundaries(heatmap)
    if isinstance(heatmap, np.ma.MaskedArray):
        sns.heatmap(
            heatmap, mask=heatmap.mask,
            vmin=v_min, vmax=v_max, cmap='plasma', ax=ax, annot=True, annot_kws={"size": 6}
        )
    else:
        sns.heatmap(heatmap, vmin=v_min, vmax=v_max, cmap='plasma', ax=ax, annot=True)


def calculate_heatmap_value_boundaries(arr: np.ndarray) -> tuple[float, float]:
    v_min, v_max = np.min(arr), np.max(arr)
    if -1 <= v_min < 0:
        v_min = -1
    elif v_min >= 0:
        v_min = 0

    if v_max < 0:
        v_max = 0
    elif v_max < 1:
        v_max = 1
    return v_min, v_max
