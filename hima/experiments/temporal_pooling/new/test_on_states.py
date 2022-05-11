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

from copy import copy

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
from hima.modules.htm.temporal_memory import ClassicTemporalMemory
from hima.modules.htm.temporal_memory import ClassicApicalTemporalMemory
from hima.experiments.temporal_pooling.test_on_policies import resolve_tp

from hima.experiments.temporal_pooling.metrics import sdrs_similarity

from hima.experiments.temporal_pooling.test_on_policies import ExperimentStats, similarity_cmp

def similarity_matrix(representations: list):
    matrix = np.empty((len(representations), len(representations)))
    for i, representation1 in enumerate(representations):
        for j, representation2 in enumerate(representations):
            matrix[i][j] = sdrs_similarity(representation1, representation2)
    return matrix


class ObservationsExperiment(Runner):
    epochs: int

    stats: ExperimentStats

    def __init__(self, config: TConfig,
                 temporal_pooler: str,
                 epochs: int,
                 rotations_per_room,
                 **kwargs):
        super().__init__(config, **config)
        self.temporal_pooler = temporal_pooler
        self.epochs = epochs
        self.rotations_per_room = rotations_per_room
        print('==> Init')
        self.data_generator = resolve_data_generator(config)
        self.temporal_memory = ClassicApicalTemporalMemory(**config['apical_temporal_memory'])
        self.temporal_pooler = resolve_tp(config, temporal_pooler, temporal_memory=self.temporal_memory)

        self.stats = ExperimentStats(self.temporal_pooler)

    def run(self):
        print('==> Generate observations')
        observations = self.data_generator.generate_data()

        print('==> Run')
        representations = []
        for epoch in range(self.epochs):
            representations = self.train_epoch(observations)
        sim_matrix = similarity_matrix(representations)
        self.log_summary(self.data_generator.true_similarities(), sim_matrix)
        print('<==')

    def log_summary(self, input_similarity_matrix, output_similarity_matrix):
        non_diag_mask = np.logical_not(np.identity(input_similarity_matrix.shape[0]))
        diff = np.abs(input_similarity_matrix - output_similarity_matrix)
        mae = np.mean(diff[non_diag_mask])
        self.logger.summary['mae'] = mae

        self.logger.log({
            'similarities': wandb.Image(similarity_cmp(input_similarity_matrix, output_similarity_matrix))
        })


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
