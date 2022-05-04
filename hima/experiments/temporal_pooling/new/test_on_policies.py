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
from hima.experiments.temporal_pooling.new.test_on_policies_stats import ExperimentStats

from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
from hima.modules.htm.spatial_pooler import UnionTemporalPooler
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


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
            steps_per_policy: int, temporal_pooler: str, **_
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

        self.stats.on_finish(
            policies=policies,
            logger=self.logger
        )
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
        base_config_tp['upper_sp_conf']['seed'] = seed
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
