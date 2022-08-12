#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from abc import ABC, abstractmethod
from typing import Optional

from hima.common.config_utils import TConfig, resolve_value
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds


class Block(ABC):
    family: str = "base_block"
    id: int
    name: str

    sds: dict[str, Sds]

    # TODO: log to charts, what to log?

    output_sdr: SparseSdr

    def __init__(self, id_: int, name: str):
        self.id = id_
        self.name = name
        self.sds = {}

    def resolve_sds(self, name: str, sds: Sds) -> Sds:
        if name in self.sds:
            sds = resolve_value(self.sds[name], substitute_with=sds)
        self.sds[name] = sds
        return sds

    @property
    def feedforward_sds(self):
        return self.sds['feedforward']

    @property
    def output_sds(self):
        return self.sds['output']

    @property
    def context_sds(self):
        return self.sds['context']

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError()

    @property
    def tag(self):
        return f'{self.id}_{self.family}'


# class InputConnection:
#     block: Block
#     source: str
#     sds: Optional[Sds]
#
#     def __init__(self, block: Block, source: str, sds: Optional[Sds] = None):
#         self.block = block
#         self.source = source
#         self.sds = sds
#
#
# class DestinationBlock


class BlocksConnection:
    src_block: Block
    src_stream: str

    dst_block: Block
    dst_stream: str

    sds: Sds

    def __init__(self, src: TConfig, dst: TConfig, block_registry: dict[str, Block]):
        self.src_block = block_registry[src["block"]]
        self.src_stream = src["what"]
        self.sds = src['sds']
        self.dst_block = block_registry[dst["block"]]
        self.dst_stream = dst["where"]

    def align_dimensions(self):
        sds = self.src_block.sds[self.src_stream]
        sds = self.dst_block.resolve_sds(self.dst_stream, sds=sds)
        self.src_block.resolve_sds(self.src_stream, sds=sds)
