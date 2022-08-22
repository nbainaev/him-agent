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
from hima.common.utils import isnone


class Block(ABC):
    family: str = "base_block"
    id: int
    name: str

    sds: dict[str, Sds]

    # TODO: log to charts, what to log?

    output_sdr: SparseSdr

    def __init__(
            self, id_: int, name: str, *,
            requires: list[str] = None, exposes: list[str] = None
    ):
        self.id = id_
        self.name = name
        self.sds = {}

    def register_sds(self, name: str):
        self.resolve_sds(name, sds=get_unresolved_value())

    def resolve_sds(self, name: str, sds: Sds) -> Sds:
        """Resolve sds value and add it to the block's sds dictionary."""
        if name in self.sds:
            sds = resolve_value(self.sds[name], substitute_with=sds)

        self.sds[name] = Sds.as_sds(sds)
        return sds

    @abstractmethod
    def build(self, **kwargs):
        """Build block when all its configurable parameters are resolved."""
        raise NotImplementedError()

    @property
    def tag(self):
        return f'{self.id}_{self.family}'

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

    def __repr__(self):
        return f'{self.tag} {self.name}'


class Stream:
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
    block: Block
    connections: list[Pipe]

    def __init__(self, connections: list[Pipe]):
        self.connections = connections
        self.block = connections[0].dst.block

    def __repr__(self):
        if len(self.connections) == 1:
            return f'{self.connections[0]}'
        else:
            return f'{self.connections}'


class Pipeline:
    units: list[ComputationUnit]

    def __init__(self, units: list[ComputationUnit]):
        self.units = units

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
