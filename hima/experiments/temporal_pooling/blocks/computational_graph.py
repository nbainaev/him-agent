#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from abc import ABC, abstractmethod
from typing import Union

from hima.common.config_utils import resolve_value, is_resolved_value, get_unresolved_value
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class Block(ABC):
    """Base building block of the computational graph / neural network."""

    family: str = "base_block"
    id: int
    name: str

    sds: dict[str, Sds]
    sdr: dict[str, SparseSdr]

    # TODO: log to charts, what to log?

    def __init__(self, id_: int, name: str):
        self.id = id_
        self.name = name
        self.sds = {}
        self.sdr = {}

    @abstractmethod
    def build(self, **kwargs):
        """Build block when all its configurable parameters are resolved."""
        raise NotImplementedError()

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        """Make a computation given the provided input data streams."""
        raise NotImplementedError()

    def request(self, stream: str):
        """Request a value from the block's output data stream."""
        return self.sdr[stream]

    # --------------- SDS interface helpers ---------------

    def register_sds(self, name: str):
        self.resolve_sds(name, sds=get_unresolved_value())

    def register_sdr(self, name: str):
        self.sdr[name] = []

    def resolve_sds(self, name: str, sds: Sds) -> Sds:
        """Resolve sds value and add it to the block's sds dictionary."""
        if name in self.sds:
            sds = resolve_value(self.sds[name], substitute_with=sds)

        self.sds[name] = Sds.as_sds(sds)
        return sds

    # --------------- Common streams ---------------

    @property
    def feedforward_sds(self):
        return self.sds['feedforward']

    @property
    def output_sds(self):
        return self.sds['output']

    @property
    def context_sds(self):
        return self.sds['context']

    # --------------- String representation ---------------

    @property
    def tag(self):
        return f'{self.id}_{self.family}'

    def __repr__(self):
        return f'{self.tag} {self.name}'


class Stream:
    """Stream defines the named dataflow to or from a block."""

    name: str
    block: Block

    def __init__(self, name: str, block: Block):
        self.block = block
        self.name = name

        # register the stream
        self.block.register_sds(self.name)

    def resolve_sds(self, sds: Sds):
        return self.block.resolve_sds(self.name, sds)

    @property
    def sds(self):
        return self.block.sds[self.name]

    def __repr__(self):
        return f'{self.block.name}.{self.name}'


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
        self.block.compute(input_data, **kwargs)

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

    def __init__(self, units: list[ComputationUnit], block_registry: dict[str, Block]):
        self.units = units
        self.blocks = block_registry
        self.entry_block = self.blocks[list(self.blocks.keys())[0]]

    def step(self, input_data, **kwargs):
        self.entry_block.compute(input_data)

        for unit in self.units:
            unit.compute(**kwargs)

    def resolve_dimensions(self, max_iters: int = 100) -> bool:
        all_dimensions_resolved = False
        for i in range(max_iters):
            if all_dimensions_resolved:
                return True

            all_dimensions_resolved = True
            for unit in self.units:
                for pipe in unit.connections:
                    all_dimensions_resolved &= pipe.align_dimensions()
        return False

    def __repr__(self):
        return f'{self.units}'
