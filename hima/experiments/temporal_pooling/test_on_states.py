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


class ObservationsExperiment(Runner):
    temporal_pooler: str
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
