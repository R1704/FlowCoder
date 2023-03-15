class Tokenizer:
    def __init__(self):

        self.dictionary = {}
        self.reverse_dictionary = {}

        self.__add_token('<PAD>')
        self.__add_token('<UNK>')
        self.__add_token('<BOS>')
        self.__add_token('<EOS>')

    def __add_token(self, token):
        if token not in self.dictionary:
            self.dictionary[token] = len(self.dictionary)
            self.reverse_dictionary[self.dictionary[token]] = token

    def tokenize(self, string):
        return [token for token in string]

    def detokenize(self, tokens):
        return ''.join(tokens)

    def encode(self, string):
        tokens = self.tokenize(string)
        return [self.dictionary[token] for token in tokens]

    def decode(self, tokens):
        return [self.reverse_dictionary[token] for token in tokens]

    def __len__(self):
        return len(self.dictionary)
