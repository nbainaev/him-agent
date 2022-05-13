#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from htm.bindings.sdr import SDR

from hima.common.config_utils import extracted_type
from hima.common.sdr import SparseSdr, DenseSdr
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
from hima.experiments.temporal_pooling.custom_utp import CustomUtp
from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
from hima.modules.htm.spatial_pooler import UnionTemporalPooler
from hima.modules.htm.temporal_memory import ClassicApicalTemporalMemory


# noinspection PyAttributeOutsideInit
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
        active_cells: SDR = temporal_memory.get_active_cells()
        predicted_cells: SDR = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.sparse.size, active_cells.sparse.size)

        return {
            'tm/recall': recall
        }


def similarity_cmp(input_similarity_matrix, output_similarity_matrix):
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(131)
    ax1.set_title('output', size=40)
    ax2 = fig.add_subplot(132)
    ax2.set_title('input', size=40)
    ax3 = fig.add_subplot(133)
    ax3.set_title('diff', size=40)

    sns.heatmap(output_similarity_matrix, vmin=0, vmax=1, cmap='plasma', ax=ax1)
    sns.heatmap(input_similarity_matrix, vmin=0, vmax=1, cmap='plasma', ax=ax2)

    sns.heatmap(
        np.abs(output_similarity_matrix - input_similarity_matrix),
        vmin=0, vmax=1, cmap='plasma', ax=ax3, annot=True
    )
    return plt.gca()


def resolve_tp(config, temporal_pooler: str, temporal_memory):
    base_config_tp = config['temporal_poolers'][temporal_pooler]
    seed = config['seed']
    input_size = temporal_memory.columns * temporal_memory.cells_per_column if not \
        isinstance(temporal_memory, ClassicApicalTemporalMemory)  \
        else temporal_memory.columns

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
