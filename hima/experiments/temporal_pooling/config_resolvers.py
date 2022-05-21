#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config_utils import (
    extracted_type, resolve_init_params, extracted,
    resolve_absolute_quantity
)
from hima.common.sds import Sds
from hima.modules.htm.temporal_memory import DelayedFeedbackTM


def resolve_tp(tp_config, feedforward_sds: Sds, output_sds: Sds, seed: int):
    tp_config, tp_type = extracted_type(tp_config)

    if tp_type == 'UnionTp':
        from hima.modules.htm.spatial_pooler import UnionTemporalPooler
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size, seed=seed
        )
        tp = UnionTemporalPooler(**tp_config)

    elif tp_type == 'AblationUtp':
        from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape, localAreaDensity=output_sds.sparsity,
            columnDimensions=output_sds.shape, maxUnionActivity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size, seed=seed
        )
        tp = AblationUtp(**tp_config)

    elif tp_type == 'CustomUtp':
        from hima.experiments.temporal_pooling.custom_utp import CustomUtp
        tp_config = resolve_init_params(
            tp_config,
            inputDimensions=feedforward_sds.shape,
            columnDimensions=output_sds.shape, union_sdr_sparsity=output_sds.sparsity,
            seed=seed
        )
        tp = CustomUtp(**tp_config)

    elif tp_type == 'SandwichTp':
        from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
        tp_config = resolve_init_params(tp_config, seed=seed)
        tp_config['lower_sp_conf'] = resolve_init_params(
            tp_config['lower_sp_conf'],
            inputDimensions=feedforward_sds.shape,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
            potentialRadius=feedforward_sds.size
        )
        tp_config['upper_sp_conf'] = resolve_init_params(
            tp_config['upper_sp_conf'],
            inputDimensions=output_sds.shape,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
            potentialRadius=output_sds.size
        )
        tp = SandwichTp(**tp_config)

    else:
        raise KeyError(f'Temporal Pooler type "{tp_type}" is not supported')

    from hima.experiments.temporal_pooling.blocks.tp import TemporalPoolerBlock
    tp_block = TemporalPoolerBlock(feedforward_sds=feedforward_sds, output_sds=output_sds, tp=tp)
    return tp_block


def resolve_context_tm(
        tm_config, ff_sds: Sds, bc_sds: Sds, seed: int
):
    # resolve only what is available already
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        ff_sds=ff_sds, bc_sds=bc_sds, seed=seed
    )
    tm_config, ff_sds, bc_sds, bc_config = extracted(tm_config, 'ff_sds', 'bc_sds', 'basal_context')

    # resolve quantities based on FF and BC SDS settings
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        columns=ff_sds.size, context_cells=bc_sds.size,
    )

    # append extracted and resolved BC config
    tm_config |= resolve_tm_connections_region(bc_config, bc_sds, '_basal')

    from hima.experiments.temporal_pooling.blocks.context_tm import ContextTemporalMemoryBlock
    return ContextTemporalMemoryBlock(
        ff_sds=ff_sds, bc_sds=bc_sds, **tm_config
    )


def resolve_context_tm_apical_feedback(fb_sds: Sds, tm_block):
    tm_config = tm_block.tm_config

    # resolve FB SDS setting
    tm_config = resolve_init_params(
        tm_config, raise_if_not_resolved=False,
        fb_sds=fb_sds
    )
    tm_config, fb_sds, fb_config = extracted(tm_config, 'fb_sds', 'apical_feedback')

    # resolve quantities based on FB SDS settings; implicitly asserts all other fields are resolved
    tm_config = resolve_init_params(tm_config, feedback_cells=fb_sds.size)

    # append extracted and resolved FB config
    tm_config |= resolve_tm_connections_region(fb_config, fb_sds, '_apical')

    tm_block.configure_apical_feedback(fb_sds=fb_sds, resolved_tm_config=tm_config)


def resolve_tm_connections_region(connections_config, sds, suffix):
    sample_size = sds.active_size
    activation_threshold = resolve_absolute_quantity(
        connections_config['activation_threshold'],
        baseline=sample_size
    )
    learning_threshold = resolve_absolute_quantity(
        connections_config['learning_threshold'],
        baseline=sample_size
    )
    max_synapses_per_segment = resolve_absolute_quantity(
        connections_config['max_synapses_per_segment'],
        baseline=sample_size
    )
    induced_config = dict(
        sample_size=sample_size,
        activation_threshold=activation_threshold,
        learning_threshold=learning_threshold,
        max_synapses_per_segment=max_synapses_per_segment
    )
    connections_config = connections_config | induced_config
    return {
        f'{k}{suffix}': connections_config[k]
        for k in connections_config
    }


def resolve_run_setup(config: dict, run_setup_config):
    if isinstance(run_setup_config, str):
        run_setup_config = config['run_setups'][run_setup_config]

    from hima.experiments.temporal_pooling.new.test_on_policies import RunSetup
    return RunSetup(**run_setup_config)


def resolve_data_generator(config: dict, **induction_registry):
    generator_config, generator_type = extracted_type(config['generator'])

    if generator_type == 'synthetic':
        from hima.experiments.temporal_pooling.data_generation import SyntheticGenerator
        generator_config = resolve_init_params(generator_config, **induction_registry)
        return SyntheticGenerator(config, **generator_config)
    elif generator_type == 'aai_rotation':
        from hima.experiments.temporal_pooling.data_generation import AAIRotationsGenerator
        return AAIRotationsGenerator(**generator_config)
    else:
        raise KeyError(f'{generator_type} is not supported')


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
