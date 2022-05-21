#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb

from hima.envs.biogwlab.env import BioGwLabEnvironment
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR
from hima.modules.motivation import Amygdala, StriatumBlock, Policy
from hima.common.run_utils import Runner
from hima.common.config_utils import TConfig
from hima.envs.biogwlab.utils.state_provider import GwAgentStateProvider
from hima.common.sdr import SparseSdr
from hima.envs.biogwlab.module import EntityType
from hima.envs.env import unwrap
from hima.envs.biogwlab.environment import Environment
from hima.modules.empowerment import Empowerment


def xlogx(x):
    mask = x != 0
    y = x[mask]
    return np.sum(y * np.log2(y))


class SDRMetrics:
    def __init__(self, sdr_size):
        self.sdr_frequency = {}
        self.sdr_bit_frequency = {}
        self.bit_frequency = np.zeros(sdr_size)

        self.total_sdrs = 0
        self.sdr_size = sdr_size
        self.n_activations = 0

    def add(self, key, value):
        if key in self.sdr_frequency.keys():
            self.sdr_frequency[key] += 1
        else:
            self.sdr_frequency[key] = 1
            self.sdr_bit_frequency[key] = np.zeros(self.sdr_size)

        self.sdr_bit_frequency[key][value] += 1
        self.bit_frequency[value] += 1
        self.total_sdrs += 1
        self.n_activations += len(value)

    @property
    def sparsity(self):
        return self.n_activations / (self.sdr_size * self.total_sdrs)

    @property
    def entropy_stability(self):
        stability = 0
        for key in self.sdr_bit_frequency.keys():
            p = self.sdr_bit_frequency[key] / self.sdr_frequency[key]
            e = - xlogx(p) - xlogx(1 - p)
            n = np.sum(self.sdr_bit_frequency[key] != 0)
            stability += e / n
        return stability / len(self.sdr_bit_frequency)

    @property
    def sdr_entropy(self):
        p = np.zeros(len(self.sdr_frequency))
        for ind, key in enumerate(self.sdr_frequency.keys()):
            p[ind] = self.sdr_frequency[key]
        p = p / np.sum(p)
        return - xlogx(p)

    @property
    def max_sdr_entropy(self):
        n = len(self.sdr_frequency)
        return np.log2(n)

    @property
    def rel_sdr_entropy(self):
        return self.sdr_entropy / self.max_sdr_entropy

    @property
    def bit_entropy(self):
        p = self.bit_frequency / self.total_sdrs
        return - xlogx(p) - xlogx(1 - p)

    @property
    def max_bit_entropy(self):
        return self.sdr_size

    @property
    def rel_bit_entropy(self):
        return self.bit_entropy / self.max_bit_entropy

    @property
    def max_sdr_per_full_entropy(self):
        s = self.sparsity
        e = - s * np.log2(s) - (1 - s) * np.log2(1 - s)
        return e

    @property
    def redundancy(self):
        e = self.bit_entropy
        E = self.sdr_entropy
        return (e - E) / E


class SPMetrics(SDRMetrics):
    def __init__(self, size):
        super().__init__(size)
        self.sdr_codes = {}
        self.pre_stability = 0

    def reset(self):
        self.n_activations = 0
        self.pre_stability = 0
        self.total_sdrs = 0
        self.bit_frequency = np.zeros(self.sdr_size)

    def add(self, key, value):
        if key in self.sdr_frequency.keys():
            self.pre_stability += len(np.intersect1d(
                value, self.sdr_codes[key])
            ) / len(self.sdr_codes[key])
            self.sdr_codes[key] = np.copy(value)
            self.sdr_frequency[key] += 1
        else:
            self.sdr_frequency[key] = 1
            self.sdr_bit_frequency[key] = np.zeros(self.sdr_size)
            self.sdr_codes[key] = np.copy(value)

        self.sdr_bit_frequency[key][value] += 1
        self.bit_frequency[value] += 1
        self.total_sdrs += 1
        self.n_activations += len(value)

    @property
    def stability(self):
        return self.pre_stability / self.total_sdrs


class GwMotivationRunner(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self._rng = np.random.default_rng(self.seed)

        self.n_episodes = config['n_episodes']
        self.evaluate_step = config['evaluate_step']
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image})
        else:
            plt.imshow(map_image)
            plt.show()
        print(f"Environment sdr size: {self.environment.output_sdr_size}")

        state_space_size = config['state_space_size']
        self.sp = SpatialPooler(
            seed=self.seed, inputDimensions=[self.environment.output_sdr_size],
            columnDimensions=[state_space_size], **config['sp'])
        self.sp_input = SDR(self.environment.output_sdr_size)
        self.sp_output = SDR(state_space_size)

        self.emp = Empowerment(
            seed=self.seed, encode_size=state_space_size,
            sparsity=self.sp.getLocalAreaDensity(), **config['emp']
        )

        self.prev_state = None
        self.episode = 0
        self.steps = 0
        self.metrics = SDRMetrics(self.environment.output_sdr_size)
        self.state_metrics = SPMetrics(state_space_size)

    def get_all_observations(self) -> dict[tuple[int, int], SparseSdr]:
        height, width = self.environment.shape
        obstacle_mask = self.environment.aggregated_mask[EntityType.Obstacle]
        position_provider = GwAgentStateProvider(self.environment)
        encoding_scheme = {}

        for i in range(height):
            for j in range(width):
                if obstacle_mask[i, j]:
                    continue
                position = i, j
                position_provider.overwrite(position)
                obs = self.environment.render()
                encoding_scheme[position] = obs

        position_provider.restore()
        return encoding_scheme

    def log_metrics(self):
        observations = self.get_all_observations()
        for key in observations.keys():
            self.sp_input.sparse = observations[key]
            self.sp.compute(self.sp_input, learn=False, output=self.sp_output)
            self.state_metrics.add(key, self.sp_output.sparse)

        anomaly = np.mean(self.emp.anomalies[-self.steps+1:])
        self.logger.log(
            {
                'anomaly': anomaly,
                'steps': self.steps,
                'obs/sdr_entropy': self.metrics.rel_sdr_entropy,
                'obs/bit_entropy': self.metrics.rel_bit_entropy,
                'obs/sparsity': self.metrics.sparsity,
                'obs/entropy_stability': self.metrics.entropy_stability,
                'obs/max_sdr_per_full_entropy': self.metrics.max_sdr_per_full_entropy,
                'state/sdr_entropy': self.state_metrics.rel_sdr_entropy,
                'state/bit_entropy': self.state_metrics.rel_bit_entropy,
                'state/sparsity': self.state_metrics.sparsity,
                'state/stability': self.state_metrics.stability,
                'state/entropy_stability': self.state_metrics.entropy_stability,
                'state/max_sdr_per_full_entropy': self.state_metrics.max_sdr_per_full_entropy,
            }, step=self.episode
        )
        self.state_metrics.reset()

    def run(self):
        self.episode = 0
        self.steps = 0

        while True:

            reward, obs, is_first = self.environment.observe()
            self.metrics.add(self.environment.agent.position, obs)
            self.sp_input.sparse = obs
            self.sp.compute(self.sp_input, learn=True, output=self.sp_output)
            if self.prev_state is None:
                self.prev_state = np.copy(self.sp_output.sparse)
            else:
                self.emp.learn(self.prev_state, self.sp_output.sparse)
                self.prev_state = np.copy(self.sp_output.sparse)

            if is_first:
                if self.episode != 0 and self.logger and self.episode % self.evaluate_step == 0:
                    self.log_metrics()
                self.episode += 1
                self.steps = 0
                if self.episode > self.n_episodes:
                    break
            else:
                self.steps += 1

            action = self._rng.integers(0, self.environment.n_actions)
            self.environment.act(action)
