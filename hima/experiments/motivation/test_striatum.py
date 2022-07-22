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

from hima.experiments.motivation.test_empowerment import plot_valued_map
from hima.agents.motivation.agent import Agent


class GwStriatumTest(Runner):
    def __init__(self, config: TConfig, **kwargs):
        super().__init__(config, **config)

        self.seed = config['seed']
        self.n_episodes = config['n_episodes']
        self.environment: Environment = unwrap(BioGwLabEnvironment(**config['environment']))
        map_image = self.environment.callmethod('render_rgb')
        if isinstance(map_image, list):
            map_image = map_image[0]
        if self.logger:
            map_image = wandb.Image(map_image)
            self.logger.log({'map': map_image}, step=0)
        else:
            plt.imshow(map_image)
            plt.show()

        self.agent = Agent(
            self.environment.output_sdr_size, self.environment.n_actions, config['agent_config']
        )

        self.prev_state = None
        self.episode = 0
        self.steps = 0

        self.visit_map = np.ma.zeros(self.environment.shape)
        self.visit_map[:, :] = np.ma.masked

    def log_metrics(self):
        self.logger.log(
            {
                'steps': self.steps,
                'maps/visit_map': wandb.Image(plot_valued_map(self.visit_map, ''))
            }, step=self.episode
        )

    def run(self):
        while True:
            reward, obs, is_first = self.environment.observe()
            if self.visit_map.mask[self.environment.agent.position]:
                self.visit_map[self.environment.agent.position] = 1
            else:
                self.visit_map[self.environment.agent.position] += 1

            if is_first:
                if self.episode != 0 and self.logger:
                    self.log_metrics()
                self.episode += 1
                self.steps = 0
                if self.episode > self.n_episodes:
                    break
            else:
                self.steps += 1

            action = self.agent.act(obs, reward, is_first)
            self.environment.act(action)
