#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import argparse
import os
from copy import deepcopy
from multiprocessing import Process

import wandb
from wandb.sdk.wandb_run import Run

from hima.common.config_utils import (
    extracted, read_config, override_config, parse_arg,
    TConfigOverrideKV
)
from hima.common.utils import isnone


class Runner:
    config: dict
    logger: Run

    def __init__(self, config: dict):
        self.config = config

        if self.config.get('log', False):
            wandb_project = self.config['project']
            self.logger = wandb.init(project=wandb_project)
            # we have to pass the config with update instead of init because of sweep runs
            self.logger.config.update(self.config)

    def run(self) -> None:
        print('==> SPAWNED')
        ...
        print('<== DESPAWNED')


class Sweep:
    id: str
    project: str
    config: dict
    n_agents: int
    agents_shared_config: dict
    agents_shared_config_overrides: list[TConfigOverrideKV]

    def __init__(
            self, config: dict, n_agents: int,
            shared_config_overrides: list[TConfigOverrideKV]
    ):
        config, run_command_args, wandb_project = extracted(
            config, 'command', 'project'
        )

        self.config = config
        self.n_agents = isnone(n_agents, 1)
        self.project = wandb_project

        shared_config_filepath = self._extract_agents_shared_config_filepath(run_command_args)
        self.agents_shared_config = read_config(shared_config_filepath)
        self.agents_shared_config_overrides = shared_config_overrides

        self.id = wandb.sweep(self.config, project=wandb_project)

    def run(self):
        print(f'==> Sweep {self.id}')

        agent_processes = []
        for _ in range(self.n_agents):
            p = Process(
                target=wandb.agent,
                kwargs={
                    'sweep_id': self.id,
                    'function': self.agent_entry_point
                }
            )
            p.start()
            agent_processes.append(p)

        for p in agent_processes:
            p.join()

        print(f'<== Sweep {self.id}')

    def agent_entry_point(self):
        # BE CAREFUL: this method is expected to be run in parallel — DO NOT edit `self` here

        # we know here that it's a sweep-induced run and can expect single sweep run config to be
        # passed via wandb.config, hence we take it and apply all overrides
        run = wandb.init()
        sweep_overrides = list(map(parse_arg, run.config.items()))

        # concat overrides, the order matters: run params, then args
        config_overrides = sweep_overrides + self.agents_shared_config_overrides

        # it's important to take COPY of the shared config
        config = deepcopy(self.agents_shared_config)
        override_config(config, config_overrides)

        Runner(config).run()

    @staticmethod
    def _extract_agents_shared_config_filepath(run_args):
        parser = get_arg_parser()
        args, _ = parser.parse_known_args(run_args)
        return args.config_filepath


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # todo: add examples
    # todo: remove --sweep ?
    parser.add_argument('-c', '--config', dest='config_filepath', required=True)
    parser.add_argument('-e', '--entity', dest='wandb_entity', required=False, default=None)
    parser.add_argument('--sweep', dest='wandb_sweep', action='store_true', default=False)
    parser.add_argument('-n', '--n_sweep_agents', type=int, default=None)
    return parser


def main():
    parser = get_arg_parser()
    args, unknown_args = parser.parse_known_args()

    config = read_config(args.config_filepath)
    config_overrides = list(map(parse_arg, unknown_args))

    if args.wandb_entity:
        # overwrite wandb entity for the run
        os.environ['WANDB_ENTITY'] = args.wandb_entity

    if args.wandb_sweep:
        Sweep(
            config=config,
            n_agents=args.n_sweep_agents,
            shared_config_overrides=config_overrides
        ).run()
    else:

        # run_single()
        ...


if __name__ == '__main__':
    main()
