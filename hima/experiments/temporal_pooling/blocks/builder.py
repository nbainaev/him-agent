#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config_utils import (
    extracted_family, resolve_nested_configs, TConfig
)
from hima.experiments.temporal_pooling.blocks.dataset_resolver import resolve_data_generator_new


def build_block(config: TConfig, block_name: str, **induction_registry):
    block_config = config['blocks'][block_name]

    block_config, block_family = extracted_family(block_config)
    assert block_family is not None

    family_registry = config[block_family]

    print(family_registry)
    block_config = resolve_nested_configs(family_registry, config=block_config)
    print(block_config)
    return _resolve_block(config, block_config, block_family, **induction_registry)


def _resolve_block(
        global_config: TConfig, block_config: TConfig, family: str, **induction_registry
):
    if family == 'generator':
        return resolve_data_generator_new(global_config, block_config, **induction_registry)


def build_connection(connection_config: TConfig, blocks: dict):
    pass
