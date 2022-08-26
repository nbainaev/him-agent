#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Any

from hima.common.config_utils import (
    extracted_family, resolve_nested_configs, TConfig, is_resolved_value, resolve_value, extracted
)
from hima.common.sds import Sds
from hima.common.utils import isnone
from hima.experiments.temporal_pooling.new.blocks.computational_graph import (
    Pipe, Block, Pipeline,
    ComputationUnit, Stream
)
from hima.experiments.temporal_pooling.new.blocks.dataset_resolver import resolve_data_generator
from hima.experiments.temporal_pooling.new.blocks.sp import resolve_sp


class BlockBuilder:
    config: TConfig
    induction_registry: dict[str, Any]

    def __init__(self, config: TConfig, **induction_registry):
        self.config = config
        self.induction_registry = induction_registry

    def build_all(self, blocks: dict[str, TConfig]) -> dict[str, Block]:
        return {
            block_name: self.build(block_id, block_name, blocks[block_name])
            for block_id, block_name in enumerate(blocks)
        }

    def build(self, block_id: int, block_name: str, block_config: TConfig) -> Block:
        block_config, block_family = extracted_family(block_config)
        assert block_family is not None
        family_registry = self.config[block_family]
        block_config = resolve_nested_configs(family_registry, config=block_config)

        # extract block's stream interface declaration to be registered further
        block_config, requires, exposes = extracted(block_config, 'requires', 'exposes')

        block = self._resolve_block(block_config, block_family, block_id, block_name)

        # register block's stream interface sds (it still can be expanded later during building)
        for stream in _resolve_interface(requires, default_streams=['feedforward']):
            block.register_sds(stream)
        for stream in _resolve_interface(exposes, default_streams=['output']):
            block.register_sds(stream)
            block.register_sdr(stream)

        return block

    def _resolve_block(
            self, block_config: TConfig,
            block_family: str, block_id: int, block_name: str,
    ) -> Block:
        if block_family == 'generator':
            return resolve_data_generator(
                self.config, block_config, block_id, block_name, **self.induction_registry
            )
        elif block_family == 'spatial_pooler':
            return resolve_sp(block_config, block_id, block_name, **self.induction_registry)


class PipelineResolver:
    blocks: dict[str, Block]

    _previous_block: Optional[str]
    _current_block: Optional[str]

    def __init__(self, block_registry: dict[str, Block]):
        self.blocks = block_registry
        self._current_block = None
        self._previous_block = None

    def resolve(self, pipeline: list) -> Pipeline:
        if not is_resolved_value(pipeline):
            # default: blocks chaining "output -> feedforward"
            units = []
            blocks = list(self.blocks.keys())
            src = blocks[0]
            for dst in blocks[1:]:
                units.append(ComputationUnit([
                    Pipe(
                        src=Stream('output', self.blocks[src]),
                        dst=Stream('feedforward', self.blocks[dst])
                    )
                ]))
                src = dst
        else:
            # defined with config
            units = []
            self._current_block = self._previous_block = None
            for unit in pipeline:
                unit = self.parse_unit(unit)
                units.append(unit)
                self._previous_block = unit.block.name
                self._current_block = None

        pipeline = Pipeline(units, block_registry=self.blocks)
        resolved = pipeline.resolve_dimensions()
        assert resolved, 'Cannot resolve one of the sds in pipeline!'

        for block in self.blocks:
            self.blocks[block].build()

        return pipeline

    def parse_unit(self, unit) -> ComputationUnit:
        def _parse_unit(
                pipe: str = None, *, sds: Sds = None,
                block: str = None, pipes: list[str, dict] = None
        ) -> ComputationUnit:
            if block is not None:
                # it is used to resolve pipes during parsing
                self._current_block = block

            pipes = isnone(pipes, [pipe])
            connections = []
            for pipe in pipes:
                if isinstance(pipe, dict):
                    pipe = self.parse_pipe(**pipe)
                else:
                    pipe = self.parse_pipe(pipe, sds=sds)
                connections.append(pipe)

            return ComputationUnit(connections=connections)

        if isinstance(unit, dict):
            return _parse_unit(**unit)
        elif isinstance(unit, str):
            return _parse_unit(pipe=unit)

    def parse_pipe(self, pipe: str, *, sds: Any = None) -> Pipe:
        src, dst = pipe.strip().split('->')
        src = self.parse_stream(src, default_block=self._previous_block)
        dst = self.parse_stream(dst, default_block=self._current_block)
        pipe = Pipe(src=src, dst=dst, sds=sds)
        return pipe

    def parse_stream(self, s: str, default_block: str = None) -> Stream:
        stream_parts = s.strip().split('.')
        if len(stream_parts) == 1:
            # default block with the given stream name
            block_name = default_block
            stream_name = stream_parts[0]
        elif len(stream_parts) == 2:
            block_name, stream_name = stream_parts
            # resolve with default if needed
            block_name = resolve_value(block_name, substitute_with=default_block)
        else:
            raise ValueError(f'Cannot parse stream from "{s}"')

        block = self.blocks[block_name]
        return Stream(name=stream_name, block=block)


def _resolve_interface(streams: list[str], default_streams: list[str]) -> list[str]:
    # None means default, unresolved means do nothing it is inducted later
    streams = resolve_value(streams)
    streams = isnone(streams, default_streams)

    return streams if is_resolved_value(streams) else []
