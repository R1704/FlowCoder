import src.sequential.deepsynth.dsl as dsl
from src.sequential.deepsynth.DSL.list import *
from src.sequential.deepsynth.Predictions.dataset_sampler import Dataset
from src.sequential.deepsynth.Predictions.embeddings import RNNEmbedding
from src.sequential.deepsynth.Predictions.IOencodings import FixedSizeEncoding
from src.sequential.deepsynth.type_system import *


import torch

class Data:
    def __init__(self, device, dataset_size, nb_examples_max, max_program_depth, nb_arguments_max, lexicon, size_max, embedding_output_dimension, number_layers_RNN, size_hidden):
        self.device = device
        self.dataset_size = dataset_size
        self.nb_examples_max = nb_examples_max
        self.max_program_depth = max_program_depth
        self.nb_arguments_max = nb_arguments_max
        self.lexicon = lexicon
        self.size_max = size_max
        self.embedding_output_dimension = embedding_output_dimension
        self.number_layers_RNN = number_layers_RNN
        self.size_hidden = size_hidden
        self.dsl = dsl.DSL(semantics, primitive_types)
        self.type_request = Arrow(List(INT), List(INT))
        self.create_cfg()
        self.create_IOEncoder_and_IOEmbedder()
        self.create_latent_encoder()
        self.create_dataset_and_generator()

    def __block__(self, input_dim, output_dimension, activation):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dimension),
            activation,
        )

    def create_cfg(self):
        self.cfg = self.dsl.DSL_to_CFG(
            self.type_request, max_program_depth=self.max_program_depth)


    def create_IOEncoder_and_IOEmbedder(self):
        self.IOEncoder = FixedSizeEncoding(
            nb_arguments_max=self.nb_arguments_max,
            lexicon=self.lexicon,
            size_max=self.size_max,
        )

        self.IOEmbedder = RNNEmbedding(
            IOEncoder=self.IOEncoder,
            output_dimension=self.embedding_output_dimension,
            size_hidden=self.size_hidden,
            number_layers_RNN=self.number_layers_RNN,
        )

    def create_latent_encoder(self):
        self.latent_encoder = torch.nn.Sequential(
            self.__block__(self.IOEncoder.output_dimension * self.IOEmbedder.output_dimension, self.size_hidden, torch.nn.Sigmoid()),
            self.__block__(self.size_hidden, self.size_hidden, torch.nn.Sigmoid()),
        )

    def create_dataset_and_generator(self):
        self.dataset = Dataset(
            size=self.dataset_size,
            dsl=self.dsl,
            pcfg_dict={self.type_request: self.cfg.CFG_to_Uniform_PCFG()},
            nb_examples_max=self.nb_examples_max,
            arguments={self.type_request: self.type_request.arguments()},
            ProgramEncoder=lambda x: x,
            size_max=self.IOEncoder.size_max,
            lexicon=self.IOEncoder.lexicon[:-2],
            for_flashfill=False
        )
        self.batch_generator = self.dataset.__iter__()

    def get_next_batch(self, batch_size):
        batch_IOs, batch_program, batch_requests = [], [], []
        for _ in range(batch_size):
            io, prog, _, req = next(self.batch_generator)
            batch_IOs.append(io)
            batch_program.append(prog)
            batch_requests.append(req)

        embedded = self.IOEmbedder.forward(batch_IOs)
        latent = self.latent_encoder(embedded)
        return batch_IOs, batch_program, latent.to(self.device)
