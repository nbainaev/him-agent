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

from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
from hima.modules.htm.spatial_pooler import UnionTemporalPooler
from hima.modules.htm.temporal_memory import DelayedFeedbackTM
from hima.modules.htm.temporal_memory import ClassicTemporalMemory
from hima.experiments.temporal_pooling.test_on_policies import resolve_tp

from hima.experiments.temporal_pooling.metrics import sdrs_similarity


def similarity_matrix(representations: list):
    matrix = np.empty((len(representations), len(representations)))
    for i, representation1 in enumerate(representations):
        for j, representation2 in enumerate(representations):
            matrix[i][j] = sdrs_similarity(representation1, representation2)
    return matrix


class ObservationsExperiment(Runner):
    epochs: int

    def __init__(self, config: TConfig,
                 temporal_pooler: str,
                 epochs: int,
                 **kwargs):
        super().__init__(config, **config)
        self.temporal_pooler = temporal_pooler
        self.epochs = epochs
        print('==> Init')
        self.data_generator = resolve_data_generator(config)
        self.temporal_memory = ClassicTemporalMemory(**config['temporal_memory'])
        self.temporal_pooler = resolve_tp(config, temporal_pooler, temporal_memory=self.temporal_memory)

    def run(self):
        print('==> Generate observations')
        observations = self.data_generator.generate_data()

        print('==> Run')
        representations = []
        for epoch in range(self.epochs):
            representations = self.train_epoch(observations)
        sim_matrix = similarity_matrix(representations)
        sns.heatmap(sim_matrix, vmin=0, vmax=1, cmap='plasma')
        self.logger.summary['similarities'] = wandb.Image(plt.gca())
        print('<==')

    def train_epoch(self, observations):
        representations = []

        for room_observations in observations:
            self.temporal_pooler.reset()
            self.run_room(room_observations, learn=True)
            representations.append(self.temporal_pooler.getUnionSDR())
        return representations

    def run_room(self, room_observations, learn=True):
        tm, tp = self.temporal_memory, self.temporal_pooler

        for observation in room_observations:
            self.compute_tm_step(
                feedforward_input=observation,
                learn=learn
            )
            self.compute_tp_step(
                active_input=tm.getActiveCells(),
                predicted_input=tm.getCorrectrlyPredictedCells(),
                learn=learn
            )

    def compute_tm_step(self, feedforward_input, learn=True):
        self.temporal_memory.compute(feedforward_input, learn=learn)
        self.temporal_memory.activateDendrites(learn=learn)

    def compute_tp_step(self, active_input, predicted_input, learn=True):
        self.temporal_pooler.compute(active_input, predicted_input, learn)
