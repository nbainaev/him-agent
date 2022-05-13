#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional

from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.common.run_utils import Runner
from hima.common.sdr import SparseSdr
from hima.experiments.temporal_pooling.config_resolvers import (
    resolve_tp, resolve_tm,
    resolve_data_generator
)
from hima.experiments.temporal_pooling.new.test_on_policies_stats import ExperimentStats


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


