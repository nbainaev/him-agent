#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import matplotlib.pyplot as plt
import wandb

from hima.envs.biogwlab.env import BioGwLabEnvironment
from hima.envs.biogwlab.environment import Environment
from hima.envs.env import unwrap

from hima.common.run_utils import Runner
from hima.common.config_utils import TConfig
from hima.common.utils import softmax

from hima.modules.motivation import Striatum
from hima.common.plot_utils import transform_fig_to_image


class GwStriatumTest(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self.n_episodes = config['n_episodes']
        self._rng = np.random.default_rng(self.seed)
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image}, step=0)

        self.str = Striatum(
            self.seed, self.environment.output_sdr_size, 50,
            **config['striatum']
        )
        self.temperature = config['temperature']

        self.episode = 0
        self.steps = 0
        self.q_map = np.ma.zeros((*self.environment.shape, self.environment.n_actions))
        self.q_map[:, :] = np.ma.masked
        self.v_map = np.ma.zeros(self.environment.shape)
        self.v_map[:, :] = np.ma.masked

    def run(self):
        motiv = np.arange(10)
        while True:
            _, obs, _ = self.environment.observe()

            action = self.str.compute(obs, motiv)
            sa = self.str.state_weights.xa2sdr(obs, action)
            v = self.str.state_weights.dopa_weights.get_value(sa)
            self.v_map[self.environment.agent.position] = v

            size = self.str.action_size // self.environment.n_actions
            actions = np.array([
                len(np.intersect1d(action, np.arange(size * i, size * (i+1))))
                for i in range(self.environment.n_actions)
            ])
            p = softmax(actions, self.temperature)
            self.q_map[
                self.environment.agent.position[0],
                self.environment.agent.position[1],
                :
            ] = p
            a = self._rng.choice(self.environment.n_actions, 1, p=p)[0]

            self.environment.act(a)
            self.steps += 1
            reward, _, _ = self.environment.observe()

            action = np.intersect1d(action, np.arange(size * a, size * (a+1)))
            self.str.update(obs, motiv, action, reward)
            if self.environment.is_terminal():
                self.environment.act(a)
                self.str.reset()
                self.episode += 1
                if self.logger:
                    self.log_metrics()
                    self.logger.log({'steps': self.steps}, step=self.episode)
                self.steps = 0
                if self.episode == self.n_episodes:
                    break

    def log_metrics(self):
        base_vectors = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ]) / 2

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 10)
        fig.set_dpi(300)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(self.q_map.max(axis=-1))

        h, w, _ = self.q_map.shape
        for i in range(h):
            for j in range(w):
                y, x = i, j
                values = self.q_map[i, j]
                m = np.argmax(values)
                d = values.reshape((-1, 1)) * base_vectors
                for dx, dy in d:
                    if dx > 1e-2 or dy > 1e-2:
                        ax.arrow(x, y, dx, dy, width=0.05, color='red', length_includes_head=True)
                ax.arrow(x, y, *(d[m]), width=0.05, color='black', length_includes_head=True)

        img = transform_fig_to_image(fig)
        plt.close(fig)
        self.logger.log(
            {
                'maps/q_map': wandb.Image(img),
                'maps/v_map': wandb.Image(plt.imshow(self.v_map))
            }, step=self.episode
        )

