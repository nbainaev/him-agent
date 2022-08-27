#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from typing import Optional, Any

import numpy as np
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig, resolve_value
from hima.common.run_utils import Runner
from hima.common.sds import Sds
from hima.common.utils import timed
from hima.experiments.temporal_pooling.config_resolvers import (
    resolve_run_setup
)
from hima.experiments.temporal_pooling.new.blocks.builder import (
    PipelineResolver,
    BlockRegistryResolver
)
from hima.experiments.temporal_pooling.new.blocks.computational_graph import Block, Pipeline
from hima.experiments.temporal_pooling.new.blocks.dataset_synth_sequences import Sequence
from hima.experiments.temporal_pooling.new.stats_config import StatsMetricsConfig
from hima.experiments.temporal_pooling.new.test_stats import (
    ExperimentStats,
    RunProgress
)


class RunSetup:
    n_sequences: int
    steps_per_sequence: Optional[int]
    sequence_repeats: int
    epochs: int
    log_repeat_schedule: int
    log_epoch_schedule: int

    tp_output_sds: Sds
    sp_output_sds: Sds

    def __init__(
            self, n_sequences: int, steps_per_sequence: Optional[int],
            sequence_repeats: int, epochs: int, total_repeats: int,
            tp_output_sds: Sds.TShortNotation, sp_output_sds: Sds.TShortNotation,
            log_repeat_schedule: int = 1, log_epoch_schedule: int = 1
    ):
        self.n_sequences = n_sequences
        self.steps_per_sequence = steps_per_sequence
        self.sequence_repeats, self.epochs = resolve_epoch_runs(
            sequence_repeats, epochs, total_repeats
        )
        self.log_repeat_schedule = log_repeat_schedule
        self.log_epoch_schedule = log_epoch_schedule

        self.tp_output_sds = Sds(short_notation=tp_output_sds)
        self.sp_output_sds = Sds(short_notation=sp_output_sds)


class ObservationsLayeredExperiment(Runner):
    config: TConfig
    logger: Optional[Run]

    seed: int
    run_setup: RunSetup
    stats_config: StatsMetricsConfig
    pipeline: Pipeline
    blocks: dict[str, Block]
    progress: RunProgress
    stats: ExperimentStats

    normalization_unbias: str

    def __init__(
            self, config: TConfig, seed: int, debug: bool,
            blocks: dict[str, TConfig], pipeline: list,
            run_setup: TConfig, stats_and_metrics: TConfig,
            **_
    ):
        super().__init__(config, **config)
        print('==> Init')

        self.seed = resolve_random_seed(seed)
        self.run_setup = resolve_run_setup(config, run_setup, experiment_type='layered')
        self.stats_config = StatsMetricsConfig(**stats_and_metrics)

        self.pipeline = PipelineResolver(
            block_registry=BlockRegistryResolver(
                config=config, block_configs=blocks,
                seed=seed,
                debug=debug,
                stats_config=self.stats_config,
                n_sequences=self.run_setup.n_sequences,
            )
        ).resolve(pipeline)
        print(self.pipeline.blocks)
        print(self.pipeline)

        # self.input_data = self.pipeline.entry_block
        # self.progress = RunProgress()
        # self.stats = ExperimentStats(
        #     n_sequences=self.run_setup.n_sequences,
        #     progress=self.progress, logger=self.logger, blocks=self.blocks,
        #     stats_config=self.stats_config,
        #     debug=debug
        # )

    def run(self):
        print('==> Run')
        # define_metrics(self.logger, self.blocks)
        #
        # for epoch in range(self.run_setup.epochs):
        #     _, elapsed_time = self.train_epoch()
        #     print(f'Epoch {epoch}: {elapsed_time}')
        print('<==')

    @timed
    def train_epoch(self):
        self.progress.next_epoch()
        self.stats.on_new_epoch()

        # noinspection PyTypeChecker
        for sequence in self.input_data:
            for i_repeat in range(self.run_setup.sequence_repeats):
                self.run_sequence(sequence, i_repeat, learn=True)

        epoch_final_log_scheduled = scheduled(
            i=self.progress.epoch, schedule=self.run_setup.log_epoch_schedule,
            always_report_first=True, always_report_last=True, i_max=self.run_setup.epochs
        )
        self.stats.on_finish(epoch_final_log_scheduled)

    def run_sequence(self, sequence: Sequence, i_repeat: int = 0, learn=True):
        self.reset_blocks(block_type='temporal_memory')
        self.reset_blocks(block_type='temporal_pooler')

        log_scheduled = scheduled(
            i=i_repeat, schedule=self.run_setup.log_repeat_schedule,
            always_report_first=True, always_report_last=True, i_max=self.run_setup.sequence_repeats
        )
        self.stats.on_new_sequence(sequence.id, log_scheduled)

        for input_data in sequence:
            self.pipeline.step(input_data, learn=learn)
            self.stats.on_step()

    def reset_blocks(self, block_type):
        for name in self.blocks:
            block = self.blocks[name]
            if block.family == block_type:
                self.blocks[name].reset()


def define_metrics(logger, blocks: dict[str, Any]):
    if not logger:
        return

    logger.define_metric('epoch')
    for k in blocks:
        block = blocks[k]
        logger.define_metric(f'{block.tag}/epoch/*', step_metric='epoch')


def resolve_random_seed(seed: Optional[int]) -> int:
    seed = resolve_value(seed)
    if seed is None:
        # randomly generate a seed
        return np.random.default_rng().integers(10000)
    return seed


def resolve_epoch_runs(intra_epoch_repeats: int, epochs: int, total_repeats: int):
    total_repeats = resolve_value(total_repeats)
    intra_epoch_repeats = resolve_value(intra_epoch_repeats)
    epochs = resolve_value(epochs)
    if intra_epoch_repeats is None:
        intra_epoch_repeats = total_repeats // epochs
    if epochs is None:
        epochs = total_repeats // intra_epoch_repeats
    return intra_epoch_repeats, epochs


def scheduled(
        i: int, schedule: int = 1,
        always_report_first: bool = True, always_report_last: bool = True, i_max: int = None
):
    if always_report_first and i == 0:
        return True
    if always_report_last and i + 1 == i_max:
        return True
    if (i + 1) % schedule == 0:
        return True
    return False
