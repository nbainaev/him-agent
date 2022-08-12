#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config_utils import (
    extracted_family, resolve_nested_configs, TConfig
)
from hima.experiments.temporal_pooling.blocks.base_block import BlocksConnection, Block
from hima.experiments.temporal_pooling.blocks.dataset_resolver import resolve_data_generator_new
from hima.experiments.temporal_pooling.blocks.sp import resolve_sp_new


def build_block(config: TConfig, block_id: int, block_name: str, **induction_registry) -> Block:
    block_config = config['blocks'][block_name]

    block_config, block_family = extracted_family(block_config)
    assert block_family is not None

    family_registry = config[block_family]

    block_config = resolve_nested_configs(family_registry, config=block_config)
    return _resolve_block(
        config, block_config, block_family, block_id, block_name, **induction_registry
    )


def _resolve_block(
        global_config: TConfig, block_config: TConfig,
        family: str, block_id: int, block_name: str,
        **induction_registry
) -> Block:
    if family == 'generator':
        return resolve_data_generator_new(
            global_config, block_config, block_id, block_name, **induction_registry
        )
    elif family == 'spatial_pooler':
        return resolve_sp_new(block_config, block_id, block_name, **induction_registry)


def build_connection(connection_config: TConfig, blocks: dict):
    connection = BlocksConnection(block_registry=blocks, **connection_config)
    connection.align_dimensions()
    print(connection.src_stream, connection.dst_stream)
