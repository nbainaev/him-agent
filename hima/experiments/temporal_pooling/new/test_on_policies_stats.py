#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional

import numpy as np
import wandb
from matplotlib import pyplot as plt
import seaborn as sns
from wandb.sdk.wandb_run import Run

from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.utils import safe_divide


# noinspection PyAttributeOutsideInit
from hima.experiments.temporal_pooling.metrics import mean_absolute_error, entropy


class ExperimentStats:
    # current policy id
    policy_id: Optional[int]

    tp_expected_active_size: int
    tp_output_sdr_size: int

    last_representations: dict[int, SparseSdr]
    tp_current_representation: set
    tp_output_distribution_counts: dict[int, DenseSdr]
    tp_sequence_total_trials: dict[int, int]

    def __init__(self, temporal_pooler):
        self.policy_id = None
        self.last_representations = {}
        self.tp_current_representation = set()
        self.tp_output_distribution_counts = {}
        self.tp_output_sdr_size = temporal_pooler.output_sdr_size
        self.tp_expected_active_size = temporal_pooler.n_active_bits
        self.tp_sequence_total_trials = {}

    def on_policy_change(self, policy_id):
        self.policy_id = policy_id
        self.window_size = 1
        self.window_error = 0
        self.whole_active = None
        self.policy_repeat = 0
        self.intra_policy_step = 0

        if policy_id not in self.tp_output_distribution_counts:
            self.tp_output_distribution_counts[policy_id] = np.zeros(
                self.tp_output_sdr_size, dtype=int
            )
            self.tp_sequence_total_trials[policy_id] = 0

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

    def on_finish(self, policies, logger: Run):
        if not logger:
            return

        to_log, to_sum = self._get_summary_old_actions(policies)
        to_log2, to_sum2 = self._get_summary_new_sdrs(policies)

        to_log = to_log | to_log2
        to_sum = to_sum | to_sum2

        to_log |= self._get_final_representations()

        logger.log(to_log)
        for key, val in to_sum.items():
            logger.summary[key] = val

    # noinspection PyProtectedMember
    def _get_tp_metrics(self, temporal_pooler) -> dict:
        prev_repr = self.tp_current_representation
        curr_repr_lst = temporal_pooler.getUnionSDR().sparse
        curr_repr = set(curr_repr_lst)
        self.tp_current_representation = curr_repr
        # noinspection PyTypeChecker
        self.last_representations[self.policy_id] = curr_repr

        self.tp_sequence_total_trials[self.policy_id] += 1
        cluster_trials = self.tp_sequence_total_trials[self.policy_id]

        output_distribution_counts = self.tp_output_distribution_counts[self.policy_id]
        output_distribution_counts[curr_repr_lst] += 1
        cluster_size = np.count_nonzero(output_distribution_counts)
        cluster_distribution = output_distribution_counts / cluster_trials

        step_sparsity = safe_divide(
            len(curr_repr), self.tp_output_sdr_size
        )
        step_relative_sparsity = safe_divide(
            len(curr_repr), self.tp_expected_active_size
        )
        new_cells_ratio = safe_divide(
            len(curr_repr - prev_repr), self.tp_expected_active_size
        )
        sym_diff_cells_ratio = safe_divide(
            len(curr_repr ^ prev_repr),
            len(curr_repr | prev_repr)
        )
        step_metrics = {
            'tp/step/sparsity': step_sparsity,
            'tp/step/relative_sparsity': step_relative_sparsity,
            'tp/step/new_cells_ratio': new_cells_ratio,
            'tp/step/sym_diff_cells_ratio': sym_diff_cells_ratio,
        }

        cluster_sparsity = safe_divide(
            cluster_size, self.tp_output_sdr_size
        )
        cluster_relative_sparsity = safe_divide(
            cluster_size, self.tp_expected_active_size
        )
        cluster_binary_active_coverage = safe_divide(
            len(curr_repr), cluster_size
        )
        cluster_distribution_active_coverage = cluster_distribution[curr_repr_lst].sum()
        cluster_entropy = entropy(cluster_distribution)
        cluster_entropy_active_coverage = safe_divide(
            entropy(cluster_distribution[curr_repr_lst]),
            cluster_entropy
        )
        sequence_metrics = {
            'tp/sequence/sparsity': cluster_sparsity,
            'tp/sequence/relative_sparsity': cluster_relative_sparsity,
            'tp/sequence/cluster_binary_coverage': cluster_binary_active_coverage,
            'tp/sequence/cluster_distribution_coverage': cluster_distribution_active_coverage,
            'tp/sequence/entropy': cluster_entropy,
            'tp/sequence/entropy_coverage': cluster_entropy_active_coverage,
        }
        return step_metrics | sequence_metrics

    def _get_tm_metrics(self, temporal_memory) -> dict:
        active_cells: np.ndarray = temporal_memory.get_active_cells()
        predicted_cells: np.ndarray = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.size, active_cells.size)

        return {
            'tm/recall': recall
        }

    def _get_summary_new_sdrs(self, policies):
        n_policies = len(policies)
        diag_mask = np.identity(n_policies, dtype=bool)

        input_similarity_matrix = self._get_input_similarity(policies)
        input_similarity_matrix = np.ma.array(input_similarity_matrix, mask=diag_mask)

        output_similarity_matrix = self._get_output_similarity(self.last_representations)
        output_similarity_matrix = np.ma.array(output_similarity_matrix, mask=diag_mask)

        input_similarity_matrix = standardize_distr(input_similarity_matrix)
        output_similarity_matrix = standardize_distr(output_similarity_matrix)

        smae = mean_absolute_error(input_similarity_matrix, output_similarity_matrix)

        representation_similarity_plot = self._plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.ma.abs(output_similarity_matrix - input_similarity_matrix)
        )
        to_log = {
            'representations_similarity_sdr': representation_similarity_plot,
        }
        to_sum = {
            'standardized_mae_sdr': smae,
        }
        return to_log, to_sum

    def _get_summary_old_actions(self, policies):
        n_policies = len(policies)
        non_diag_mask = np.logical_not(np.identity(n_policies))

        input_similarity_matrix = self._get_policy_action_similarity(policies)
        output_similarity_matrix = self._get_output_similarity_union(
            self.last_representations
        )
        input_similarity_matrix[non_diag_mask] = standardize_distr(
            input_similarity_matrix[non_diag_mask]
        )
        output_similarity_matrix[non_diag_mask] = standardize_distr(
            output_similarity_matrix[non_diag_mask]
        )

        diag_mask = np.identity(n_policies, dtype=bool)
        input_similarity_matrix = np.ma.array(input_similarity_matrix, mask=diag_mask)
        output_similarity_matrix = np.ma.array(output_similarity_matrix, mask=diag_mask)

        smae = mean_absolute_error(
            input_similarity_matrix,
            output_similarity_matrix
        )

        representation_similarity_plot = self._plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.ma.abs(output_similarity_matrix - input_similarity_matrix)
        )
        to_log = {
            'representations_similarity': representation_similarity_plot,
        }
        to_sum = {
            'standardized_mae': smae,
        }
        return to_log, to_sum

    def _get_policy_action_similarity(self, policies):
        n_policies = len(policies)
        similarity_matrix = np.zeros((n_policies, n_policies))

        for i in range(n_policies):
            for j in range(n_policies):

                counter = 0
                size = 0
                for p1, p2 in zip(policies[i], policies[j]):
                    _, a1 = p1
                    _, a2 = p2

                    size += 1
                    # such comparison works only for bucket encoding
                    if a1[0] == a2[0]:
                        counter += 1

                similarity_matrix[i, j] = safe_divide(counter, size)
        return similarity_matrix

    def _get_input_similarity(self, policies):
        def elem_sim(x1, x2):
            overlap = np.intersect1d(x1, x2, assume_unique=True).size
            return safe_divide(overlap, x2.size)

        def reduce_elementwise_similarity(similarities):
            return np.mean(similarities)

        n_policies = len(policies)
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                similarities = []
                for p1, p2 in zip(policies[i], policies[j]):
                    p1_sim = [elem_sim(p1[k], p2[k]) for k in range(len(p1))]
                    sim = reduce_elementwise_similarity(p1_sim)
                    similarities.append(sim)

                similarity_matrix[i, j] = reduce_elementwise_similarity(similarities)
        return similarity_matrix

    def _get_output_similarity_union(self, representations):
        n_policies = len(representations.keys())
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                repr1: set = representations[i]
                repr2: set = representations[j]
                similarity_matrix[i, j] = safe_divide(
                    len(repr1 & repr2),
                    len(repr2 | repr2)
                )
        return similarity_matrix

    def _get_output_similarity(self, representations):
        n_policies = len(representations.keys())
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                repr1: set = representations[i]
                repr2: set = representations[j]
                similarity_matrix[i, j] = safe_divide(
                    len(repr1 & repr2),
                    len(repr2)
                )
        return similarity_matrix

    def _plot_similarity_matrices(self, **sim_matrices):
        n = len(sim_matrices)
        heatmap_size = 6
        fig, axes = plt.subplots(
            nrows=1, ncols=n, sharey='all',
            figsize=(heatmap_size * n, heatmap_size)
        )

        for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
            if isinstance(sim_matrix, np.ma.MaskedArray):
                sns.heatmap(
                    sim_matrix, mask=sim_matrix.mask,
                    vmin=-1, vmax=1, cmap='plasma', ax=ax, annot=True
                )
            else:
                sns.heatmap(sim_matrix, vmin=-1, vmax=1, cmap='plasma', ax=ax, annot=True)
            ax.set_title(name, size=10)

        return wandb.Image(axes[0])

    def _get_final_representations(self):
        n_clusters = len(self.last_representations)
        representations = np.zeros((n_clusters, self.tp_output_sdr_size), dtype=float)
        distributions = np.zeros_like(representations)

        for i, policy_id in enumerate(self.last_representations.keys()):
            repr = self.last_representations[policy_id]
            distr = self.tp_output_distribution_counts[policy_id]
            trials = self.tp_sequence_total_trials[policy_id]

            representations[i, list(repr)] = 1.
            distributions[i] = distr / trials

        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        sns.heatmap(representations, vmin=0, vmax=1, cmap='plasma')

        fig, ax2 = plt.subplots(1, 1, figsize=(16, 8))
        sns.heatmap(distributions, vmin=0, vmax=1, cmap='plasma', ax=ax2)

        return {
            'representations': wandb.Image(ax1),
            'distributions': wandb.Image(ax2)
        }


def standardize_distr(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)
