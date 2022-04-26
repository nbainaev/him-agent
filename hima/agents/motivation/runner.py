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
from hima.modules.motivation import Amygdala, StriatumBlock
from hima.common.run_utils import Runner
from hima.common.config_utils import TConfig


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

        print('==> Environment')
        self.n_episodes = config['n_episodes']
        self.evaluate_step = config['evaluate_step']
        self.environment = BioGwLabEnvironment(**config['environment'])

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

        print('==> Amygdala')
        self.amg = Amygdala(
            sdr_size=self.environment.output_sdr_size,
            seed=self.seed, **config['amygdala']
        )

        print('==> Striatum')
        self.str_amg = StriatumBlock(
            inputDimensions=self.amg.out_sdr_shape,
            seed=self.seed, **config['striatum']
        )
        self.str_sma = StriatumBlock(
            inputDimensions=[1, self.environment.output_sdr_size],
            seed=self.seed, **config['striatum']
        )
        self.striatum_output_sdr_size = self.str_sma.output_sdr_size + self.str_amg.output_sdr_size

        self.episode = 0
        self.steps = 0
        self.metrics = SDRMetrics(self.environment.output_sdr_size)
        self.str_metrics = SPMetrics(self.striatum_output_sdr_size)
        self.base_states = self.create_base_states()

    def create_base_states(self):
        base_states = dict()
        env = deepcopy(self.environment)
        for i in range(env.env.shape[0]):
            for j in range(env.env.shape[1]):
                if not env.env.entities['obstacle'].mask[i, j]:
                    env.env.agent.position = (i, j)
                    _, s, _ = env.observe()
                    sdr_new = SDR(env.output_sdr_size)
                    sdr_new.sparse = s
                    base_states[(i, j)] = sdr_new
        return base_states

    def run(self):
        print('==> Run')
        self.episode = 0
        self.steps = 0
        self.counter = 0

        while True:

            reward, obs, is_first = self.environment.observe()
            self.counter += 1

            sdr_amg = self.str_amg.compute(self.amg.compute(obs), 0, True)
            sdr_sma = self.str_sma.compute(obs, 0, True)
            sdr_str = np.concatenate((
                sdr_amg, self.str_amg.output_sdr_size + sdr_sma
            ))

            self.metrics.add(self.environment.env.agent.position, obs)

            if self.counter % self.evaluate_step == 0:
                value_map = np.zeros(self.environment.env.shape)
                for key in self.base_states.keys():
                    value_map[key] = self.amg.get_value(self.base_states[key].sparse)
                    sdr_amg = self.str_amg.compute(self.amg.compute(self.base_states[key].sparse), 0, False)
                    sdr_sma = self.str_sma.compute(self.base_states[key].sparse, 0, False)
                    sdr_str = np.concatenate((
                        sdr_amg, self.str_amg.output_sdr_size + sdr_sma
                    ))
                    self.str_metrics.add(key, sdr_str)
                if self.logger:
                    self.logger.log({
                        'value_map': wandb.Image(plt.imshow(value_map)),
                        'env/sdr_entropy': self.metrics.rel_sdr_entropy,
                        'env/bit_entropy': self.metrics.rel_bit_entropy,
                        'env/redundancy': self.metrics.redundancy,
                        'env/sparsity': self.metrics.sparsity,
                        'env/entropy_stability': self.metrics.entropy_stability,
                        'env/max_sdr_per_full_entropy': self.metrics.max_sdr_per_full_entropy,
                        'striatum/sdr_entropy': self.str_metrics.rel_sdr_entropy,
                        'striatum/bit_entropy': self.str_metrics.rel_bit_entropy,
                        'striatum/redundancy': self.str_metrics.redundancy,
                        'striatum/sparsity': self.str_metrics.sparsity,
                        'striatum/stability': self.str_metrics.stability,
                        'striatum/entropy_stability': self.str_metrics.entropy_stability,
                        'striatum/max_sdr_per_full_entropy': self.str_metrics.max_sdr_per_full_entropy,
                    })
                self.str_metrics.reset()

            if is_first:
                # plt.imshow(self.amg.value_network.cell_value.reshape((9, 9)))
                # plt.show()
                self.episode += 1
                self.steps = 0
                self.amg.reset()
                self.str_sma.reset()
                self.str_amg.reset()
                if self.episode > self.n_episodes:
                    break
            else:
                self.steps += 1

            self.amg.update(obs, reward)
            action = self._rng.integers(0, self.environment.n_actions)
            self.environment.act(action)

        print('<==')