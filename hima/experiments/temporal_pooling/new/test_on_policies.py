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
from hima.common.utils import isnone, timed
from hima.experiments.temporal_pooling.blocks.policies_dataset import Policy
from hima.experiments.temporal_pooling.config_resolvers import (
    resolve_tp, resolve_data_generator, resolve_run_setup, resolve_context_tm,
    resolve_context_tm_apical_feedback
)
from hima.experiments.temporal_pooling.new.test_on_policies_stats import (
    ExperimentStats,
    RunProgress
)


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
    progress: RunProgress
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
        self.progress = RunProgress()

    def run(self):
        print('==> Run')
        self.define_metrics(self.logger, self.blocks)

        for epoch in range(self.run_setup.epochs):
            _, elapsed_time = self.train_epoch()
            print(f'Epoch {epoch}: {elapsed_time}')
        print('<==')

    @timed
    def train_epoch(self):
        self.progress.next_epoch()
        self.stats = ExperimentStats(
            progress=self.progress, logger=self.logger, blocks=self.blocks
        )
        self.reset_blocks_stats()

        for policy in self.input_data:
            for i in range(self.run_setup.policy_repeats):
                self.run_policy(policy, learn=True)

        self.stats.on_finish()

    def run_policy(self, policy: Policy, learn=True):
        self.reset_blocks(block_type='temporal_memory')
        self.reset_blocks(block_type='temporal_pooler')
        self.stats.on_new_sequence(policy.id)

        for action, state in policy:
            self.step(state, action, learn)

    def step(self, state: SparseSdr, action: SparseSdr, learn: bool):
        self.progress.next_step()

        feedforward, feedback = [], []
        # context is fixed for all levels (as I don't know what another context to take)
        context = state
        prev_block = None

        for block_name in self.pipeline:
            block = self.blocks[block_name]

            if block_name == 'generator':
                output = action

            elif block_name.startswith('temporal_memory'):
                output = block.compute(
                    feedforward_input=feedforward, basal_context=context, learn=learn
                )

            else:   # temporal pooler
                goes_after_tm = prev_block.name.startswith('temporal_memory')
                if goes_after_tm:
                    active_input, correctly_predicted_input = feedforward
                    output = block.compute(
                        active_input=active_input,
                        predicted_input=correctly_predicted_input,
                        learn=learn
                    )
                    prev_block.pass_feedback(output)
                else:
                    output = block.compute(
                        active_input=feedforward, predicted_input=feedforward, learn=learn
                    )

            feedforward = output
            prev_block = block

        self.stats.on_step()

    def reset_blocks(self, block_type):
        for block_name in self.pipeline:
            if block_name.startswith(block_type):
                self.blocks[block_name].reset()

    def reset_blocks_stats(self):
        for block_name in self.pipeline:
            self.blocks[block_name].reset_stats()

    def build_blocks(self, temporal_pooler: str) -> dict:
        blocks = {}
        feedforward_sds, context_sds = None, None
        prev_block = None
        for block_ind, block_name in enumerate(self.pipeline):
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
                if prev_block.name.startswith('temporal_memory'):
                    resolve_context_tm_apical_feedback(
                        fb_sds=block.output_sds, tm_block=prev_block
                    )
                feedforward_sds = block.output_sds

            else:
                raise KeyError(f'Block name "{block_name}" is not supported')

            block.id = block_ind
            block.name = block_name
            blocks[block.name] = block
            prev_block = block

        return blocks

    @staticmethod
    def define_metrics(logger, blocks: dict[str, Any]):
        if not logger:
            return

        logger.define_metric('epoch')
        for k in blocks:
            block = blocks[k]
            logger.define_metric(f'{block.tag}/epoch/*', step_metric='epoch')
