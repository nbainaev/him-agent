#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.common.config_utils import extracted_type, resolve_init_params, TConfig
from hima.experiments.temporal_pooling.new.stats_config import StatsMetricsConfig


def resolve_data_generator(
        global_config: TConfig, generator_config: TConfig,
        block_id: int, block_name: str,
        n_sequences: int, stats_config: StatsMetricsConfig,
        **induction_registry
):
    generator_config, generator_type = extracted_type(generator_config)

    if generator_type == 'synthetic_sequences':
        from hima.experiments.temporal_pooling.new.blocks.dataset_synth_sequences import (
            SyntheticSequencesGenerator
        )
        generator_config = resolve_init_params(generator_config, **induction_registry)
        generator = SyntheticSequencesGenerator(global_config, **generator_config)
    else:
        raise KeyError(f'{generator_type} is not supported')

    sequences = generator.generate_sequences(n_sequences)
    generator_block = generator.build_block(block_id, block_name, sequences, stats_config)
    return generator_block


def resolve_encoder(
        config: dict, key, registry_key: str,
        n_values: int = None, active_size: int = None, seed: int = None
):
    registry = config[registry_key]
    encoder_config, encoder_type = extracted_type(registry[key])

    if encoder_type == 'int_bucket':
        from hima.common.sdr_encoders import IntBucketEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, bucket_size=active_size
        )
        return IntBucketEncoder(**encoder_config)
    if encoder_type == 'int_random':
        from hima.common.sdr_encoders import IntRandomEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, active_size=active_size, seed=seed
        )
        return IntRandomEncoder(**encoder_config)
    else:
        raise KeyError(f'{encoder_type} is not supported')
