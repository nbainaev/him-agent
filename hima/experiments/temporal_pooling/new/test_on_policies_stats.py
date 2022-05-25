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

from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.metrics import (
    similarity_matrix,
    standardize_sample_distribution
)
from hima.experiments.temporal_pooling.utils import rename_dict_keys


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
    progress: RunProgress
    logger: Optional[Run]
    blocks: dict[str, Any]
    sequence_ids_order: list[int]
    sequences_block_stats: dict[int, dict[str, Any]]

    def __init__(self, progress: RunProgress, logger: Optional[Run], blocks: dict[str, Any]):
        self.progress = progress
        self.logger = logger
        self.blocks = blocks
        self.sequences_block_stats = {}
        self.sequence_ids_order = []

    def on_new_sequence(self, sequence_id: int):
        if sequence_id == self.current_sequence_id:
            return
        self.sequence_ids_order.append(sequence_id)
        self.sequences_block_stats[sequence_id] = {}

        for block_name in self.blocks:
            block = self.blocks[block_name]
            block.reset_stats()
            self.current_block_stats[block.name] = block.stats

    @property
    def current_sequence_id(self):
        return self.sequence_ids_order[-1] if self.sequence_ids_order else None

    @property
    def previous_sequence_id(self):
        return self.sequence_ids_order[-2] if len(self.sequence_ids_order) >= 2 else None

    @property
    def current_block_stats(self):
        return self.sequences_block_stats[self.current_sequence_id]

    def on_step(self):
        if self.logger is None:
            return

        metrics = {
            'epoch': self.progress.epoch
        }
        for block_name in self.current_block_stats:
            block = self.blocks[block_name]
            block_stats = self.current_block_stats[block_name]
            block_metrics = block_stats.step_metrics()
            block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/')
            metrics |= block_metrics

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def on_finish(self):
        if not self.logger:
            return

        metrics = {}
        diff_metrics = []
        optimized_metrics = []
        for block_name in self.current_block_stats:
            block = self.blocks[block_name]
            if block_name.startswith('generator'):
                block_metrics, block_diff_metrics = self.summarize_input(block)
            elif block_name.startswith('temporal_memory'):
                block_metrics, block_diff_metrics = {}, {}
            else:   # temporal_pooler
                block_metrics, block_diff_metrics, repr_pmf_cov = self.summarize_tp(block)
                optimized_metrics.append(repr_pmf_cov)

            block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/epoch/')
            metrics |= block_metrics
            diff_metrics.append((block.tag, block_diff_metrics))

        metrics |= self.summarize_similarity_errors(diff_metrics, optimized_metrics)

        if self.logger:
            self.logger.log(metrics, step=self.progress.step)

    def summarize_input(self, block):
        block_stats = self.current_block_stats[block.name]
        metrics = block_stats.final_metrics()
        diff_metrics = {}
        for metric_key in metrics:
            metric_value = metrics[metric_key]

            if metric_key == 'raw_sim_mx_el':
                diff_metrics['raw_sim_mx'] = metric_value
            if metric_key == 'sim_mx_el':
                diff_metrics['sim_mx'] = metric_value

            if isinstance(metric_value, np.ndarray) and metric_value.ndim == 2:
                metrics[metric_key] = self.plot_representations(metric_value)
        return metrics, diff_metrics

    def summarize_tp(self, block):
        raw_metrics = self._collect_block_final_stats(block)
        metrics = self._sdr_representation_similarities(
            raw_metrics['representative'], block.output_sds
        )
        metrics |= self._pmf_similarities(
            raw_metrics['distribution'], block.output_sds
        )
        metrics['mean_repr_pmf_coverage'] = np.mean(raw_metrics['representative_pmf_coverage'])
        metrics['mean_relative_sparsity'] = np.mean(raw_metrics['relative_sparsity'])

        diff_metrics = {}
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if metric_key in {'raw_sim_mx', 'sim_mx'}:
                diff_metrics[metric_key] = metric_value
            if isinstance(metric_value, np.ndarray) and metric_value.ndim == 2:
                metrics[metric_key] = self.plot_representations(metric_value)
        return metrics, diff_metrics, metrics['mean_repr_pmf_coverage']

    def summarize_similarity_errors(self, diff_metrics, optimized_metrics):
        input_tag, input_sims = diff_metrics[0]
        metrics = {
            sim_key: {input_tag: input_sims[sim_key]}
            for sim_key in input_sims
        }

        discount, pmf_alpha = .85, 0.1
        gamma, loss = 1, 0

        for i in range(1, len(diff_metrics)):
            block_tag, block_sim_metrics = diff_metrics[i]
            pmf_coverage = optimized_metrics[i]

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
                    loss += gamma * (mae + pmf_alpha * (1 - pmf_coverage) / pmf_coverage)

            gamma *= discount

        result = {}
        for metric_key in metrics.keys():
            metric = metrics[metric_key]
            if isinstance(metric, dict):
                # dict of similarity matrices
                result[f'diff/{metric_key}'] = self._plot_similarity_matrices(**metric)
            else:
                result[metric_key] = metric

        result['loss'] = loss
        return result

    @staticmethod
    def _plot_similarity_matrices(**sim_matrices):
        n = len(sim_matrices)
        heatmap_size = 6
        fig, axes = plt.subplots(
            nrows=1, ncols=n, sharey='all',
            figsize=(heatmap_size * n, heatmap_size)
        )

        for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
            vmin = 0 if np.min(sim_matrix) >= 0 else -1
            if isinstance(sim_matrix, np.ma.MaskedArray):
                sns.heatmap(
                    sim_matrix, mask=sim_matrix.mask,
                    vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True
                )
            else:
                sns.heatmap(sim_matrix, vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True)
            ax.set_title(name, size=10)

        img = wandb.Image(axes[0])
        plt.close(fig)
        return img

    @staticmethod
    def _sdr_representation_similarities(representations, sds: Sds):
        raw_similarity_matrix = similarity_matrix(representations, symmetrical=False, sds=sds)
        raw_similarity = raw_similarity_matrix.mean()
        stand_similarity_matrix = standardize_sample_distribution(raw_similarity_matrix)

        return {
            'raw_sim_mx': raw_similarity_matrix,
            'raw_sim': raw_similarity,
            'sim_mx': stand_similarity_matrix,
        }

    @staticmethod
    def _pmf_similarities(representations, sds: Sds):
        raw_similarity_matrix_kl = similarity_matrix(
            representations, algorithm='kl-divergence', symmetrical=False, sds=sds
        )
        raw_similarity_kl = raw_similarity_matrix_kl.mean()
        similarity_matrix_kl = standardize_sample_distribution(raw_similarity_matrix_kl)

        raw_similarity_matrix_was = similarity_matrix(
            representations, algorithm='wasserstein', symmetrical=False, sds=sds
        )
        raw_similarity_was = raw_similarity_matrix_was.mean()
        similarity_matrix_was = standardize_sample_distribution(raw_similarity_matrix_kl)

        return {
            'raw_sim_mx_kl': raw_similarity_matrix_kl,
            'raw_sim_kl': raw_similarity_kl,
            'sim_mx_kl': similarity_matrix_kl,

            'raw_sim_mx_was': raw_similarity_matrix_was,
            'raw_sim_was': raw_similarity_was,
            'sim_mx_was': similarity_matrix_was,
        }

    def _collect_block_final_stats(self, block) -> dict[str, Any]:
        result = {}
        n_sequences = len(self.sequences_block_stats)
        for seq_ind in self.sequences_block_stats:
            block_stat = self.sequences_block_stats[seq_ind][block.name]
            final_metrics = block_stat.final_metrics()
            for metric_key in final_metrics:
                if metric_key not in result:
                    result[metric_key] = [None]*n_sequences
                result[metric_key][seq_ind] = final_metrics[metric_key]

        for metric_key in result:
            if isinstance(result[metric_key][0], np.ndarray):
                result[metric_key] = np.vstack(result[metric_key])
        return result

    @staticmethod
    def plot_representations(repr_matrix):
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        vmin = 0 if np.min(repr_matrix) >= 0 else -1
        if isinstance(repr_matrix, np.ma.MaskedArray):
            sns.heatmap(
                repr_matrix, mask=repr_matrix.mask,
                vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True
            )
        else:
            sns.heatmap(repr_matrix, vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True)
        img = wandb.Image(fig)
        plt.close(fig)
        return img
