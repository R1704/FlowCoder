"""
The tokenizer takes one or more ASTs and tokenizes them.
Here a smart tokenization is necessary as to figure out what useful program components are.
In DreamCoder this would essentially be the abstractions,
but using GFlowNet we can just use the most salient intermediate constructions.
So do we need this class to tokenize ASTs, or are they given by the GFN?

Once we collect a couple useful program components in the primitives, we can use them to tokenize the ASTs.
So we tokenize ASTs given the learnt concepts.
"""