class Tokenizer:
    def __init__(self):

        self.dictionary = {}
        self.reverse_dictionary = {}

        self.__add_token('<PAD>')   # padding token (idx = 0)
        self.__add_token('<UNK>')   # unknown token (idx = 1)
        self.__add_token('<BOS>')   # beginning of string token (idx = 2)
        self.__add_token('<EOS>')   # end of string token (idx = 3)

    def __add_token(self, token):
        if token not in self.dictionary:
            self.dictionary[token] = len(self.dictionary)
            self.reverse_dictionary[self.dictionary[token]] = token

    def tokenize(self, string):
        for c in string:
            self.__add_token(c)
        return [self.dictionary[c] for c in string]

    def character_to_token(self, character):
        return self.dictionary[character]

    def token_to_character(self, token):
        return self.reverse_dictionary[token]

    def __len__(self):
        return len(self.dictionary)
