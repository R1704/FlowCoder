import src.sequential.deepsynth.dsl as dsl
from src.sequential.deepsynth.DSL.list import *
from src.sequential.deepsynth.Predictions.dataset_sampler import Dataset
from src.sequential.deepsynth.type_system import *

import torch

from dataclasses import dataclass


@dataclass
class Data:
    device: torch
    dataset_size: int
    nb_examples_max: int
    max_program_depth: int
    nb_arguments_max: int
    lexicon: list
    size_max: int

    def __post_init__(self):
        self.dsl = dsl.DSL(semantics, primitive_types)
        self.type_request = Arrow(List(INT), List(INT))
        self.create_cfg()
        self.create_dataset_and_generator()

    def create_cfg(self):
        self.cfg = self.dsl.DSL_to_CFG(
            self.type_request, max_program_depth=self.max_program_depth)

    def create_dataset_and_generator(self):
        self.dataset = Dataset(
            size=self.dataset_size,
            dsl=self.dsl,
            pcfg_dict={self.type_request: self.cfg.CFG_to_Uniform_PCFG()},
            nb_examples_max=self.nb_examples_max,
            arguments={self.type_request: self.type_request.arguments()},
            ProgramEncoder=lambda x: x,
            size_max=self.size_max,
            lexicon=self.lexicon,
            for_flashfill=False
        )
        self.batch_generator = self.dataset.__iter__()

    # def make_new_dataset(self):
    #     self.create_dataset_and_generator()

    def get_next_batch(self, batch_size):
        batch_IOs, batch_program = [], []
        for _ in range(batch_size):
            io, prog, _, _ = next(self.batch_generator)
            batch_IOs.append(io)
            batch_program.append(prog)
        return batch_IOs, batch_program
