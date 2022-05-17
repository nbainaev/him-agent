#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Any

from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.common.run_utils import Runner
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.blocks.context_tm import ContextTemporalMemoryBlock
from hima.experiments.temporal_pooling.config_resolvers import (
    resolve_tp, resolve_data_generator, resolve_run_setup, resolve_context_tm,
    resolve_context_tm_apical_feedback
)
from hima.experiments.temporal_pooling.blocks.policies_dataset import Policy
from hima.experiments.temporal_pooling.new.test_on_policies_stats import ExperimentStats


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
    pipeline: list[str]
    blocks: dict[str, Any]
    stats: ExperimentStats

    def __init__(
            self, config: TConfig, run_setup, seed: int,
            pipeline: list[str],
            policy_selection_rule: str, temporal_pooler: str, **_
    ):
        super().__init__(config, **config)
        self.seed = seed
        self.run_setup = resolve_run_setup(config, run_setup)

        print('==> Init')
        self.pipeline = pipeline
        self.blocks = self.build_blocks(temporal_pooler)
        self.input_data = self.blocks[self.pipeline[0]]

        # self.stats = ExperimentStats(self.temporal_pooler)

    def run(self):
        print('==> Generate policies')

        print('==> Run')
        for epoch in range(self.run_setup.epochs):
            self.train_epoch()

        # self.stats.on_finish(
        #     policies=policies,
        #     logger=self.logger
        # )
        print('<==')

    def train_epoch(self):
        for policy in self.input_data:
            self.reset_blocks(block_type='temporal_pooler')
            for i in range(self.run_setup.policy_repeats):
                self.run_policy(policy, learn=True)

    def run_policy(self, policy: Policy, learn=True):
        for state, action in policy:
            self.step(state, action, learn)

    def step(self, state: SparseSdr, action: SparseSdr, learn: bool):
        feedforward, feedback = [], []
        # context is fixed for all levels (as I don't know what another context to take)
        context = state
        prev_block_name = None

        for block_name in self.pipeline:
            block = self.blocks[block_name]

            if block_name == 'generator':
                feedforward = action
                # stats
            elif block_name.startswith('temporal_memory'):
                feedforward = block.compute(
                    feedforward_input=feedforward, basal_context=context, learn=learn
                )

                active_input, correctly_predicted_input = feedforward
                # stats
            elif block_name.startswith('temporal_pooler'):
                after_tm = prev_block_name.startswith('temporal_memory')
                if after_tm:
                    active_input, correctly_predicted_input = feedforward
                    feedforward = block.compute(
                        active_input=active_input,
                        predicted_input=correctly_predicted_input,
                        learn=learn
                    )
                    self.blocks[prev_block_name].pass_feedback(feedforward)
                else:
                    feedforward = block.compute(
                        active_input=feedforward, predicted_input=feedforward, learn=learn
                    )
                # stats
            prev_block_name = block_name

        # self.stats.on_step(
        #     policy_id=policy.id,
        #     temporal_memory=self.temporal_memory,
        #     temporal_pooler=self.temporal_pooler,
        #     logger=self.logger
        # )

    def reset_blocks(self, block_type):
        for block_name in self.pipeline:
            if block_name.startswith(block_type):
                self.blocks[block_name].reset()

    def build_blocks(self, temporal_pooler: str) -> dict:
        blocks = {}
        feedforward_sds, context_sds = None, None
        prev_block_name = None
        for block_name in self.pipeline:
            if block_name == 'generator':
                data_generator = resolve_data_generator(
                    self.config,
                    n_states=self.run_setup.n_states,
                    n_actions=self.run_setup.n_actions,
                    seed=self.seed
                )
                block = data_generator.generate_policies(self.run_setup.n_policies)
                feedforward_sds = block.output_sds
                context_sds = block.context_sds
            elif block_name.startswith('temporal_memory'):
                block = resolve_context_tm(
                    tm_config=self.config['temporal_memory'],
                    ff_sds=feedforward_sds,
                    bc_sds=context_sds,
                    seed=self.seed
                )
                feedforward_sds = block.output_sds
            elif block_name.startswith('temporal_pooler'):
                block = resolve_tp(
                    self.config['temporal_poolers'][temporal_pooler],
                    feedforward_sds=feedforward_sds,
                    output_sds=self.run_setup.tp_output_sds,
                    seed=self.seed
                )
                if prev_block_name.startswith('temporal_memory'):
                    resolve_context_tm_apical_feedback(
                        fb_sds=block.output_sds, tm_block=blocks[prev_block_name]
                    )
                feedforward_sds = block.output_sds
            else:
                raise KeyError(f'Block name "{block_name}" is not supported')

            blocks[block_name] = block
            prev_block_name = block_name
        return blocks
