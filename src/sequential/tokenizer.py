import re


class Tokenizer:
    def __init__(self, grammar):
        self.grammar = grammar
        self.vocab = ['<START>', '<STOP>'] + grammar.primitives
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.terminals_pattern = '|'.join([re.escape(op) for op in self.grammar.terminals])
        self.nonterminals_pattern = '|'.join([re.escape(op) for op in self.grammar.nonterminals])

    def seq2tokens(self, seq):
        return re.findall(self.terminals_pattern + '|' + self.nonterminals_pattern + '|<START>|<STOP>', ''.join(seq))

    def encode(self, seq):
        token_seq = self.seq2tokens(seq)
        return [self.token_to_idx[token] for token in token_seq]

    def decode(self, seq):
        return [self.idx_to_token[idx] for idx in seq]
