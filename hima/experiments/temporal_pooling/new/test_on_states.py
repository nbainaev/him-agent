#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Union, Any

import numpy as np
import wandb
from htm.bindings.sdr import SDR
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import TConfig
from hima.common.run_utils import Runner
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.common.utils import timed
from hima.experiments.temporal_pooling.blocks.dataset_aai import RoomObservationSequence
from hima.experiments.temporal_pooling.blocks.dataset_resolver import resolve_data_generator
from hima.experiments.temporal_pooling.blocks.tm_sequence import (
    resolve_tm,
    resolve_tm_apical_feedback
)
from hima.experiments.temporal_pooling.blocks.tp import resolve_tp
from hima.experiments.temporal_pooling.config_resolvers import resolve_run_setup
from hima.experiments.temporal_pooling.new.test_on_states_stats import RunProgress, ExperimentStats


# def similarity_matrix(representations: list):
#     matrix = np.empty((len(representations), len(representations)))
#     for i, representation1 in enumerate(representations):
#         for j, representation2 in enumerate(representations):
#             matrix[i][j] = sdrs_similarity(representation1, representation2)
#     return matrix


class RunSetup:
    n_sequences: Optional[int]
    n_observations_per_sequence: Optional[int]
    sequence_repeats: int
    epochs: int

    tp_output_sds: Sds

    def __init__(
            self, sequence_repeats: int, epochs: int, tp_output_sds: Sds.TShortNotation,
            n_sequences: Optional[int] = None, n_observations_per_sequence: Optional[int] = None
    ):
        self.n_sequences = n_sequences
        self.n_observations_per_sequence = n_observations_per_sequence
        self.epochs = epochs
        self.sequence_repeats = sequence_repeats
        self.tp_output_sds = Sds(short_notation=tp_output_sds)


class ObservationsExperiment(Runner):
    config: TConfig
    logger: Optional[Run]

    seed: int
    run_setup: RunSetup
    pipeline: list[str]
    blocks: dict[str, Any]
    progress: RunProgress
    stats: ExperimentStats

    def __init__(
            self, config: TConfig, run_setup: Union[dict, str], seed: int,
            pipeline: list[str], temporal_pooler: str, **_
    ):
        super().__init__(config, **config)
        self.seed = seed
        self.run_setup = resolve_run_setup(config, run_setup, experiment_type='observations')

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

        for room_obs_sequence in self.input_data:
            for i in range(self.run_setup.sequence_repeats):
                self.run_sequence(room_obs_sequence, learn=True)

        self.stats.on_finish()

    def run_sequence(self, room_obs_sequence: RoomObservationSequence, learn=True):
        self.reset_blocks(block_type='temporal_memory')
        self.reset_blocks(block_type='temporal_pooler')
        self.stats.on_new_sequence(room_obs_sequence.id)

        for observation in room_obs_sequence:
            self.step(observation, learn)

    def step(self, observation: SparseSdr, learn: bool):
        self.progress.next_step()

        feedforward, feedback = [], []
        prev_block = None

        for block_name in self.pipeline:
            block = self.blocks[block_name]

            if block_name == 'generator':
                output = observation

            elif block_name.startswith('temporal_memory'):
                output = block.compute(feedforward_input=feedforward, learn=learn)

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
        feedforward_sds = None
        prev_block = None
        for block_ind, block_name in enumerate(self.pipeline):
            if block_name == 'generator':
                data_generator = resolve_data_generator(self.config)
                block = data_generator.generate_data(
                    n_sequences=self.run_setup.n_sequences,
                    n_observations_per_sequence=self.run_setup.n_observations_per_sequence
                )
                feedforward_sds = block.output_sds

            elif block_name.startswith('temporal_memory'):
                block = resolve_tm(
                    tm_config=self.config['temporal_memory'],
                    ff_sds=feedforward_sds,
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
                    resolve_tm_apical_feedback(
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
