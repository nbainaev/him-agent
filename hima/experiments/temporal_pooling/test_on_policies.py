#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig, extracted_type
from hima.common.run_utils import Runner
from hima.common.sdr import SparseSdr
from hima.common.utils import safe_divide, ensure_absolute_number
from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
from hima.experiments.temporal_pooling.custom_utp import CustomUtp
from hima.experiments.temporal_pooling.data_generation import resolve_data_generator

from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
from hima.modules.htm.spatial_pooler import UnionTemporalPooler
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


# noinspection PyAttributeOutsideInit
class ExperimentStats:
    tp_expected_active_size: int
    tp_output_sdr_size: int

    def __init__(self, temporal_pooler):
        self.policy_id: Optional[int] = None
        self.last_representations = {}
        self.tp_current_representation = set()
        # self.tp_prev_policy_union = tp.getUnionSDR().copy()
        # self.tp_prev_union = tp.getUnionSDR().copy()
        self.tp_output_distribution = {}
        self.tp_output_sdr_size = temporal_pooler.output_sdr_size
        self.tp_expected_active_size = temporal_pooler.n_active_bits

    def on_policy_change(self, policy_id, temporal_pooler):
        # self.tp_prev_policy_union = self.tp_prev_union.copy()
        self.tp_prev_union = set(temporal_pooler.getUnionSDR().sparse)

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
            self.on_policy_change(policy_id, temporal_pooler)

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
            # len(curr_repr - prev_repr), self.tp_expected_active_size
            len(curr_repr - self.tp_prev_union), self.tp_expected_active_size
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


class PoliciesExperiment(Runner):
    config: TConfig
    logger: Optional[Run]

    n_policies: int
    epochs: int
    policy_repeats: int
    steps_per_policy: int

    stats: ExperimentStats

    _tp_active_input: SDR
    _tp_predicted_input: SDR

    def __init__(
            self, config: TConfig, n_policies: int, epochs: int, policy_repeats: int,
            steps_per_policy: int, temporal_pooler: str, **kwargs
    ):
        super().__init__(config, **config)

        self.n_policies = n_policies
        self.epochs = epochs
        self.policy_repeats = policy_repeats
        # --------------------------------------
        self.steps_per_policy = steps_per_policy
        # ---- is this field really needed? ----

        print('==> Init')
        self.data_generator = resolve_data_generator(config)
        self.temporal_memory = resolve_tm(
            self.config,
            action_encoder=self.data_generator.action_encoder,
            state_encoder=self.data_generator.state_encoder
        )
        self.temporal_pooler = resolve_tp(
            self.config, temporal_pooler,
            temporal_memory=self.temporal_memory
        )
        self.stats = ExperimentStats(self.temporal_pooler)

        # pre-allocated SDR
        tp_input_size = self.temporal_pooler.getNumInputs()
        self._tp_active_input = SDR(tp_input_size)
        self._tp_predicted_input = SDR(tp_input_size)

    def run(self):
        print('==> Generate policies')
        policies = self.data_generator.generate_policies(self.n_policies)

        print('==> Run')
        for epoch in range(self.epochs):
            self.train_epoch(policies)

        self.log_summary(policies)
        print('<==')

    def train_epoch(self, policies):
        representations = []

        for policy in policies:
            self.temporal_pooler.reset()
            for i in range(self.policy_repeats):
                self.run_policy(policy, learn=True)

            representations.append(self.temporal_pooler.getUnionSDR())
        return representations

    def run_policy(self, policy, learn=True):
        tm, tp = self.temporal_memory, self.temporal_pooler

        for state, action in policy:
            self.compute_tm_step(
                feedforward_input=action,
                basal_context=state,
                apical_feedback=self.temporal_pooler.getUnionSDR().sparse,
                learn=learn
            )
            self.compute_tp_step(
                active_input=tm.get_active_cells(),
                predicted_input=tm.get_correctly_predicted_cells(),
                learn=learn
            )
            self.stats.on_step(
                policy_id=policy.id,
                temporal_memory=self.temporal_memory,
                temporal_pooler=self.temporal_pooler,
                logger=self.logger
            )

    def log_summary(self, policies):
        def centralize_sim_matrix(sim_matrix):
            # mean row sim ==> diag
            sim_matrix[diag_mask] = sim_matrix[non_diag_mask].mean(axis=-1)
            # centralize
            sim_matrix[diag_mask] -= sim_matrix[non_diag_mask].mean()

        if not self.logger:
            return

        n_policies = len(policies)
        diag_mask = np.identity(n_policies, dtype=bool)
        non_diag_mask = np.logical_not(np.identity(n_policies))

        input_similarity_matrix = self._get_policy_action_similarity(policies)
        output_similarity_matrix = self._get_output_similarity_union(
            self.stats.last_representations
        )
        centralize_sim_matrix(input_similarity_matrix)
        centralize_sim_matrix(output_similarity_matrix)
        diff = np.abs(input_similarity_matrix - output_similarity_matrix)
        mae = np.mean(diff[non_diag_mask])
        self.logger.summary['mae'] = mae
        self.vis_similarity(
            input_similarity_matrix, output_similarity_matrix, 'representations similarity'
        )

        input_similarity_matrix = self._get_input_similarity(policies)
        output_similarity_matrix = self._get_output_similarity(self.stats.last_representations)
        centralize_sim_matrix(input_similarity_matrix)
        centralize_sim_matrix(output_similarity_matrix)
        diff = np.abs(input_similarity_matrix - output_similarity_matrix)
        mae = np.mean(diff[non_diag_mask])
        self.logger.summary['mae_alt'] = mae
        self.vis_similarity(
            input_similarity_matrix, output_similarity_matrix, 'representations similarity alt'
        )

    def compute_tm_step(
            self, feedforward_input: SparseSdr, basal_context: SparseSdr,
            apical_feedback: SparseSdr, learn: bool
    ):
        tm = self.temporal_memory

        tm.set_active_context_cells(basal_context)
        tm.activate_basal_dendrites(learn)

        tm.set_active_feedback_cells(apical_feedback)
        tm.activate_apical_dendrites(learn)
        tm.propagate_feedback()

        tm.predict_cells()

        tm.set_active_columns(feedforward_input)
        tm.activate_cells(learn)

    def compute_tp_step(self, active_input: SparseSdr, predicted_input: SparseSdr, learn: bool):
        self._tp_active_input.sparse = active_input.copy()
        self._tp_predicted_input.sparse = predicted_input.copy()

        self.temporal_pooler.compute(self._tp_active_input, self._tp_predicted_input, learn)

    @staticmethod
    def _get_policy_action_similarity(policies):
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
                        counter +=1

                similarity_matrix[i, j] = safe_divide(counter, size)
        return similarity_matrix

    @staticmethod
    def _get_input_similarity(policies):
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

    @staticmethod
    def _get_output_similarity_union(representations):
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

    @staticmethod
    def _get_output_similarity(representations):
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

    def vis_similarity(self, input_similarity_matrix, output_similarity_matrix, title):
        fig = plt.figure(figsize=(40, 10))
        ax1 = fig.add_subplot(131)
        ax1.set_title('output', size=40)
        ax2 = fig.add_subplot(132)
        ax2.set_title('input', size=40)
        ax3 = fig.add_subplot(133)
        ax3.set_title('diff', size=40)

        sns.heatmap(output_similarity_matrix, vmin=-1, vmax=1, cmap='plasma', ax=ax1)
        sns.heatmap(input_similarity_matrix, vmin=-1, vmax=1, cmap='plasma', ax=ax2)

        sns.heatmap(
            np.abs(output_similarity_matrix - input_similarity_matrix),
            vmin=-1, vmax=1, cmap='plasma', ax=ax3, annot=True
        )
        self.logger.log({title: wandb.Image(ax1)})


def resolve_tp(config, temporal_pooler: str, temporal_memory):
    base_config_tp = config['temporal_poolers'][temporal_pooler]
    seed = config['seed']
    input_size = temporal_memory.columns * temporal_memory.cells_per_column

    config_tp = dict(
        inputDimensions=[input_size],
        potentialRadius=input_size,
    )

    base_config_tp, tp_type = extracted_type(base_config_tp)
    if tp_type == 'UnionTp':
        config_tp = base_config_tp | config_tp
        tp = UnionTemporalPooler(seed=seed, **config_tp)
    elif tp_type == 'AblationUtp':
        config_tp = base_config_tp | config_tp
        tp = AblationUtp(seed=seed, **config_tp)
    elif tp_type == 'CustomUtp':
        config_tp = base_config_tp | config_tp
        del config_tp['potentialRadius']
        tp = CustomUtp(seed=seed, **config_tp)
    elif tp_type == 'SandwichTp':
        # FIXME: dangerous mutations here! We should work with copies
        base_config_tp['lower_sp_conf'] = base_config_tp['lower_sp_conf'] | config_tp
        base_config_tp['lower_sp_conf']['seed'] = seed
        tp = SandwichTp(**base_config_tp)
    else:
        raise KeyError(f'Temporal Pooler type "{tp_type}" is not supported')
    return tp


def resolve_tm(config, action_encoder, state_encoder):
    base_config_tm = config['temporal_memory']
    seed = config['seed']

    # apical feedback
    apical_feedback_cells = base_config_tm['feedback_cells']
    apical_active_bits = ensure_absolute_number(
        base_config_tm['sample_size_apical'],
        baseline=apical_feedback_cells
    )
    activation_threshold_apical = ensure_absolute_number(
        base_config_tm['activation_threshold_apical'],
        baseline=apical_active_bits
    )
    learning_threshold_apical = ensure_absolute_number(
        base_config_tm['learning_threshold_apical'],
        baseline=apical_active_bits
    )

    # basal context
    basal_active_bits = state_encoder.n_active_bits

    config_tm = dict(
        columns=action_encoder.output_sdr_size,

        feedback_cells=apical_feedback_cells,
        sample_size_apical=apical_active_bits,
        activation_threshold_apical=activation_threshold_apical,
        learning_threshold_apical=learning_threshold_apical,
        max_synapses_per_segment_apical=apical_active_bits,

        context_cells=state_encoder.output_sdr_size,
        sample_size_basal=basal_active_bits,
        activation_threshold_basal=basal_active_bits,
        learning_threshold_basal=basal_active_bits,
        max_synapses_per_segment_basal=basal_active_bits,
    )

    # it's necessary as we shadow some "relative" values with the "absolute" values
    config_tm = base_config_tm | config_tm
    tm = DelayedFeedbackTM(seed=seed, **config_tm)
    return tm
