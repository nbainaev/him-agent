#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Any

from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.common.run_utils import Runner
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.config_resolvers import (
    resolve_tp, resolve_data_generator, resolve_run_setup, resolve_context_tm,
    resolve_context_tm_apical_feedback
)
from hima.experiments.temporal_pooling.new.test_on_policies_stats import ExperimentStats
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


class ContextTemporalMemoryBlock:
    feedforward_sds: Sds
    cells_per_column: int
    basal_context_sds: Sds
    apical_feedback_sds: Sds
    cells_sds: Sds

    tm: DelayedFeedbackTM
    tm_config: dict

    def __init__(self, ff_sds: Sds, bc_sds: Sds, **partially_resolved_tm_config):
        cells_per_column = partially_resolved_tm_config['cells_per_column']

        self.feedforward_sds = ff_sds
        self.cells_per_column = cells_per_column
        self.basal_context_sds = bc_sds
        self.cells_sds = Sds(
            size=self.feedforward_sds.size * cells_per_column,
            active_size=self.feedforward_sds.active_size
        )
        self.tm_config = partially_resolved_tm_config

    def set_apical_feedback(self, fb_sds, resolved_tm_config):
        self.apical_feedback_sds = fb_sds
        self.tm_config = resolved_tm_config
        self.tm = DelayedFeedbackTM(**self.tm_config)

    @property
    def output_sds(self):
        return self.cells_sds


class TemporalPoolerBlock:
    feedforward_sds: Sds
    output_sds: Sds
    tp: Any

    def __init__(self, feedforward_sds: Sds, output_sds: Sds, tp: Any):
        self.feedforward_sds = feedforward_sds
        self.output_sds = output_sds
        self.tp = tp


class RunSetup:
    n_policies: int
    n_states: int
    n_actions: int
    steps_per_policy: int
    policy_repeats: int
    epochs: int

    tp_output_sds: Sds

    def __init__(
            self, n_policies: int, n_states: int, n_actions: int,
            steps_per_policy: Optional[int], policy_repeats: int, epochs: int,
            tp_output_sds: tuple
    ):
        self.n_policies = n_policies
        self.n_states = n_states
        self.n_actions = n_actions
        self.steps_per_policy = isnone(steps_per_policy, self.n_states)
        self.epochs = epochs
        self.policy_repeats = policy_repeats
        self.tp_output_sds = Sds(tp_output_sds)


class PoliciesExperiment(Runner):
    config: TConfig
    logger: Optional[Run]

    seed: int
    run_setup: RunSetup
    pipeline: list
    stats: ExperimentStats

    _tp_active_input: SDR
    _tp_predicted_input: SDR

    def __init__(
            self, config: TConfig, run_setup, seed: int,
            pipeline: list[str],
            policy_selection_rule: str, temporal_pooler: str, **_
    ):
        super().__init__(config, **config)
        self.seed = seed
        self.run_setup = resolve_run_setup(config, run_setup)

        print('==> Init')
        self.data_generator = resolve_data_generator(
            config,
            n_states=self.run_setup.n_states,
            n_actions=self.run_setup.n_actions,
            seed=self.seed
        )

        self.pipeline = []
        for block_name in pipeline:
            if block_name == 'generator':
                block = self.data_generator
            elif block_name == 'temporal_memory':
                block = resolve_context_tm(
                    tm_config=self.config['temporal_memory'],
                    ff_sds=self.data_generator.actions_sds,
                    bc_sds=self.data_generator.states_sds,
                    seed=self.seed
                )
                print(block.feedforward_sds, block.output_sds, block.basal_context_sds)
            elif block_name == 'temporal_pooler':
                prev_block = self.pipeline[-1]
                block = resolve_tp(
                    self.config, temporal_pooler,
                    feedforward_sds=prev_block.output_sds,
                    output_sds=self.run_setup.tp_output_sds,
                    seed=seed
                )
                if isinstance(prev_block, ContextTemporalMemoryBlock):
                    resolve_context_tm_apical_feedback(block.output_sds, prev_block)
            else:
                raise KeyError(f'Block name "{block_name}" is not supported')

            self.pipeline.append(block)

        # self.temporal_pooler = resolve_tp(
        #     self.config, temporal_pooler,
        #     temporal_memory=self.temporal_memory
        # )
        # self.stats = ExperimentStats(self.temporal_pooler)
        #
        # # pre-allocated SDR
        # tp_input_size = self.temporal_pooler.getNumInputs()
        # self._tp_active_input = SDR(tp_input_size)
        # self._tp_predicted_input = SDR(tp_input_size)

    def run(self):
        print('==> Generate policies')
        return
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


