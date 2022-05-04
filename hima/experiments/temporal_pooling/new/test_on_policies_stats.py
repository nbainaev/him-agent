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
from hima.experiments.temporal_pooling.metrics import mean_absolute_error


class ExperimentStats:
    # current policy id
    policy_id: Optional[int]

    tp_expected_active_size: int
    tp_output_sdr_size: int

    last_representations: dict[int, SparseSdr]
    tp_current_representation: set
    tp_output_distribution: dict[int, DenseSdr]

    def __init__(self, temporal_pooler):
        self.policy_id = None
        self.last_representations = {}
        self.tp_current_representation = set()
        self.tp_output_distribution = {}
        self.tp_output_sdr_size = temporal_pooler.output_sdr_size
        self.tp_expected_active_size = temporal_pooler.n_active_bits

    def on_policy_change(self, policy_id):
        self.policy_id = policy_id
        self.window_size = 1
        self.window_error = 0
        self.whole_active = None
        self.policy_repeat = 0
        self.intra_policy_step = 0
        self.tp_output_distribution.setdefault(
            policy_id, np.empty(self.tp_output_sdr_size, dtype=int)
        ).fill(0)

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

    # noinspection PyProtectedMember
    def _get_tp_metrics(self, temporal_pooler) -> dict:
        prev_repr = self.tp_current_representation
        curr_repr_lst = temporal_pooler.getUnionSDR().sparse
        curr_repr = set(curr_repr_lst)
        self.tp_current_representation = curr_repr
        # noinspection PyTypeChecker
        self.last_representations[self.policy_id] = curr_repr

        output_distribution = self.tp_output_distribution[self.policy_id]
        output_distribution[curr_repr_lst] += 1

        sparsity = safe_divide(
            len(curr_repr), self.tp_expected_active_size
        )
        new_cells_ratio = safe_divide(
            len(curr_repr - prev_repr), self.tp_expected_active_size
        )
        cells_in_whole = safe_divide(
            len(curr_repr), np.count_nonzero(output_distribution)
        )
        step_difference = safe_divide(
            len(curr_repr ^ prev_repr),
            len(curr_repr | prev_repr)
        )

        return {
            'tp/sparsity': sparsity,
            'tp/new_cells': new_cells_ratio,
            'tp/cells_in_whole': cells_in_whole,
            'tp/step_diff': step_difference
        }

    def _get_tm_metrics(self, temporal_memory) -> dict:
        active_cells: np.ndarray = temporal_memory.get_active_cells()
        predicted_cells: np.ndarray = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.size, active_cells.size)

        return {
            'tm/recall': recall
        }

    def on_finish(self, policies, logger: Run):
        if not logger:
            return

        to_log, to_sum = self._get_summary_old_actions(policies)
        to_log2, to_sum2 = self._get_summary_new_sdrs(policies)

        to_log = to_log | to_log2
        to_sum = to_sum | to_sum2

        logger.log(to_log)
        for key, val in to_sum.items():
            logger.summary[key] = val

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

        representation_similarity_plot = self.plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.abs(output_similarity_matrix - input_similarity_matrix)
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

        representation_similarity_plot = self.plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.abs(output_similarity_matrix - input_similarity_matrix)
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

    def plot_similarity_matrices(self, **sim_matrices):
        n = len(sim_matrices)
        fig, axes = plt.subplots(
            nrows=1, ncols=n, sharey='all'
        )
        fig = plt.figure(figsize=(5 * n, 5))

        for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
            sns.heatmap(sim_matrix, vmin=-1, vmax=1, cmap='plasma', ax=ax)
            ax.set_title(name, size=5)

        return wandb.Image(fig)


def standardize_distr(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / np.std(x)
