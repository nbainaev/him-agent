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
from hima.common.sds import Sds
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
        self.policy_repeats = sequence_repeats
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
        return
        print('==> Generate observations')
        observations = self.data_generator.generate_data()

        print('==> Run')
        representations = []
        for epoch in range(self.epochs):
            representations = self.train_epoch(observations)
        sim_matrix = similarity_matrix(representations)
        self.log_summary(self.data_generator.true_similarities(), sim_matrix)
        print('<==')

    # def log_summary(self, input_similarity_matrix, output_similarity_matrix):
    #     non_diag_mask = np.logical_not(np.identity(input_similarity_matrix.shape[0]))
    #     diff = np.abs(input_similarity_matrix - output_similarity_matrix)
    #     mae = np.mean(diff[non_diag_mask])
    #     self.logger.summary['mae'] = mae
    #
    #     self.logger.log({
    #         'similarities': wandb.Image(similarity_cmp(input_similarity_matrix, output_similarity_matrix))
    #     })

    def train_epoch(self, observations):
        representations = []

        for i, room_observations in enumerate(observations):
            self.temporal_pooler.reset()
            for j in range(self.rotations_per_room):
                self.run_room(room_observations, i, learn=True)

            sdr = SDR(self.temporal_pooler.getNumColumns())
            sdr.sparse = self.temporal_pooler.getUnionSDR().sparse.copy()
            representations.append(sdr)
        return representations

    def run_room(self, room_observations, room_id, learn=True):
        tm, tp = self.temporal_memory, self.temporal_pooler

        for observation in room_observations:
            self.compute_tm_step(
                feedforward_input=observation,
                learn=learn
            )
            self.compute_tp_step(
                active_input=tm.get_active_columns(),
                predicted_input=tm.get_correctly_predicted_columns(),
                learn=learn
            )
            self.stats.on_step(
                policy_id=room_id,
                temporal_memory=self.temporal_memory,
                temporal_pooler=self.temporal_pooler,
                logger=self.logger
            )

    def compute_tm_step(self, feedforward_input, learn=True):
        self.temporal_memory.compute(
            activeColumns=feedforward_input.sparse,
            apicalInput=[],  # self.temporal_pooler.getUnionSDR().sparse,
            learn=learn
        )

        # self.temporal_memory.activateDendrites(learn=learn)

    def compute_tp_step(self, active_input, predicted_input, learn=True):
        self.temporal_pooler.compute(active_input, predicted_input, learn)

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
