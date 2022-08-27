#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from abc import ABC, abstractmethod
from typing import Union, Any, Optional

from hima.common.config_utils import (
    resolve_value, is_resolved_value, get_unresolved_value,
    get_none_value
)
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.stats_config import StatsMetricsConfig


class Stream:
    """Stream defines the named dataflow to or from a block."""
    name: str
    sds: Sds
    sdr: SparseSdr
    block: 'Block'
    stats: Optional['StreamStats']

    def __init__(self, name: str, block: 'Block'):
        self.block = block
        self.name = name
        self.sds = get_unresolved_value()
        self.stats = None

    def resolve_sds(self, sds: Sds) -> Sds:
        self.sds = resolve_value(self.sds, substitute_with=sds)
        self.sds = Sds.as_sds(self.sds)
        return self.sds

    def track_stats(self, stats: 'StreamStats'):
        self.stats = stats

    def put_value(self, sdr: SparseSdr):
        self.sdr = sdr
        if self.stats:
            self.stats.update(sdr)

    def __repr__(self):
        return f'{self.block.name}.{self.name}'


class Block(ABC):
    """Base building block of the computational graph / neural network."""

    family: str = "base_block"
    supported_streams: set[str] = {}

    id: int
    name: str
    streams: dict[str, Stream]

    # TODO: log to charts, what to log?

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.streams = {}

    @abstractmethod
    def build(self, **kwargs):
        """Build block when all its configurable parameters are resolved."""
        raise NotImplementedError()

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        """Make a computation given the provided input data streams."""
        raise NotImplementedError()

    def request(self, stream: str):
        """Request a value from the block's output data stream."""
        # FIXME: remove it
        raise NotImplementedError()

    # --------------- SDS interface helpers ---------------
    def register_stream(self, name: str) -> 'Stream':
        if name not in self.streams:
            self.streams[name] = Stream(name=name, block=self)
        return self.streams[name]

    def resolve_sds(self, name: str, sds: Sds) -> Sds:
        """Resolve sds value and add it to the block's sds dictionary."""
        if name in self.sds:
            sds = resolve_value(self.sds[name], substitute_with=sds)

        self.sds[name] = sds = Sds.as_sds(sds)
        return sds

    # --------------- Stats interface helpers ---------------
    def track_stats(self, name: str, stats_config: StatsMetricsConfig):
        raise NotImplementedError()

    def reset_stats(self):
        raise NotImplementedError()

    # --------------- String representation ---------------

    @property
    def tag(self):
        return f'{self.id}_{self.family}'

    def stream_tag(self, stream: str):
        return f'{self.tag}.{stream}'

    def __repr__(self):
        return f'{self.tag} {self.name}'


class StreamStats:
    stream: Stream

    def __init__(self, stream: Stream):
        self.stream = stream

    @property
    def sds(self):
        return self.stream.sds

    def update(self, **kwargs):
        ...

    def step_metrics(self) -> dict[str, Any]:
        return {}

    def aggregated_metrics(self) -> dict[str, Any]:
        ...


class Pipe:
    """Pipe connects two blocks' streams. Thus, both streams operate in the same SDS."""

    src: Stream
    dst: Stream

    # TODO: implement delay and sdr bookkeeping
    delay: int
    _sdr: Union[SparseSdr, list[SparseSdr]]

    def __init__(self, src: Stream, dst: Stream, sds: Sds = None):
        self.src = src
        self.dst = dst

        if sds is not None:
            self.src.resolve_sds(sds)

    def align_dimensions(self) -> bool:
        if is_resolved_value(self.src.sds) and is_resolved_value(self.dst.sds):
            return True

        sds = self.src.sds
        sds = self.dst.resolve_sds(sds)
        sds = self.src.resolve_sds(sds)
        return is_resolved_value(sds)

    @property
    def sds(self):
        return self.src.sds

    def __repr__(self):
        return f'{self.sds}| {self.src} -> {self.dst}'


class ComputationUnit:
    """
    Computational unit defines a node in the computational graph.
    It joins several input connections of a block to represent a single computational unit —
    connections provide the input data for a single computation inside the block.
    """

    block: Block
    connections: list[Pipe]

    def __init__(self, connections: list[Pipe]):
        self.connections = connections
        self.block = connections[0].dst.block

    def compute(self, **kwargs):
        input_data = {
            connection.dst.name: connection.src.block.request(connection.src.name)
            for connection in self.connections
        }
        return self.block.compute(input_data, **kwargs)

    def __repr__(self):
        if len(self.connections) == 1:
            return f'{self.connections[0]}'
        else:
            return f'{self.connections}'


class Pipeline:
    """
    Pipeline is the ordered traversal of the computational graph, that is it defines both —
    the graph itself and the order of the computations.
    """
    units: list[ComputationUnit]
    blocks: dict[str, Block]

    entry_block: Block

    def __init__(self, units: list[ComputationUnit], blocks: dict[str, Block]):
        self.units = units
        self.blocks = blocks
        # self.entry_block = self.blocks[list(self.blocks.keys())[0]]

    def step(self, input_data, stats_tracker, **kwargs):
        self.entry_block.compute(input_data)

        for unit in self.units:
            output_sdr = unit.compute(**kwargs)
            stats_tracker.on_block_step(unit.block, )

    def __repr__(self):
        return f'{self.units}'


class ExternalApiBlock(Block):
    INPUT = 'input'
    OUTPUT = 'output'

    family = '_ext_api_'
    supported_streams = {INPUT, OUTPUT}

    def build(self, **kwargs):
        pass

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        pass

    def request(self, stream: str):
        pass

    def track_stats(self, name: str, stats_config: StatsMetricsConfig):
        pass

    def reset_stats(self):
        pass
