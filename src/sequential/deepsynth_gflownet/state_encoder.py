import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence


class RuleEncoder(nn.Module):
    def __init__(self, cfg, d_model=512, device='cpu'):
        super(RuleEncoder, self).__init__()
        self.device = device
        self.cfg = cfg

        # Collecting rules (non-terminal to program pairs) from the CFG
        # TODO: Do I also need a STOP token,
        #  so the transformer learns what a finished program looks like?
        self.rules = ['PAD', 'START']
        self.terminal_rules = []
        for non_terminal, programs in cfg.rules.items():
            for program, args in programs.items():
                self.rules.append((non_terminal, program))
                if args == []:
                    self.terminal_rules.append((non_terminal, program))


        # Creating dictionaries for indexing
        self.rule2idx = {rule: i for i, rule in enumerate(self.rules)}
        self.idx2rule = {i: rule for rule, i in self.rule2idx.items()}

        n_rules = len(self.rules)
        self.rule_embedding = nn.Embedding(num_embeddings=n_rules, embedding_dim=d_model)

    def forward(self, states_batch):

        # Convert rules to indices and then convert lists to tensors and move them to the specified device
        states_batch = [torch.tensor([self.rule2idx[s] for s in state], device=self.device) for state in states_batch]

        # Encode the state sequences into embeddings
        states_encoded = [self.rule_embedding(state) for state in states_batch]

        # Padding
        states_encoded = pad_sequence(states_encoded, batch_first=True, padding_value=self.rule2idx['PAD'])
        states_encoded = states_encoded.transpose(0, 1)
        return states_encoded

    def get_parent_rule(self, rule):
        parents = []
        (_, (my_p, _), my_depth), _ = rule
        for r in self.rules[2:]:
            (_, _, depth), prog = r
            if prog == my_p and my_depth - 1 == depth:
                parents.append(r)
        return parents

    def get_parent_args(self, rule):
        nt, p = rule
        return self.cfg.rules[nt][p]
