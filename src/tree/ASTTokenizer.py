"""
The tokenizer takes one or more ASTs and tokenizes them.
Here a smart tokenization is necessary as to figure out what useful program components are.
In DreamCoder this would essentially be the abstractions,
but using GFlowNet we can just use the most salient intermediate constructions.
So do we need this class to tokenize ASTs, or are they given by the GFN?

Once we collect a couple useful program components in the primitives, we can use them to tokenize the ASTs.
So we tokenize ASTs given the learnt concepts.
"""
from src.tree.AST import Node

class ASTTokenizer:
    def __init__(self, current_tokens):
        self.tokens = current_tokens

    def _find_subtree(self, tree, subtree):
        if tree.is_equal_tree(subtree):
            return [tree]
        elif tree.is_terminal:
            return []
        else:
            results = []
            for child in tree.children.values():
                results.extend(self._find_subtree(child, subtree))
            return results

    def tokenize(self, tree):
        # The tokenize_recursive function takes a node and returns a list of tokenized trees
        def tokenize_recursive(node):
            # Initialize the maximum subtree size and corresponding token
            max_subtree_size = 0
            max_subtree_token = None

            # Iterate through each token in the tokenizer's list of tokens
            for token in self.tokens:
                # Check if the token is a non-terminal node
                if token.is_nonterminal:
                    # Calculate the size of the subtree rooted at this token
                    subtree_size = token.count_nodes()
                    # Find all instances of the token subtree in the input tree node
                    matches = self._find_subtree(node, token)
                    # If there are matches and the subtree is larger than the current maximum subtree
                    if len(matches) > 0 and subtree_size > max_subtree_size:
                        # Update the maximum subtree size and corresponding token
                        max_subtree_size = subtree_size
                        max_subtree_token = token

            # If no maximum subtree token is found
            if max_subtree_token is None:
                # Return the node itself if it's a terminal node, otherwise, tokenize its children
                tokenized_tree = [node]
                if not node.is_terminal:
                    for child in node.children.values():
                        tokenized_tree.extend(tokenize_recursive(child))
                return tokenized_tree
            else:
                # Tokenize children of the current node
                tokenized_children = [tokenize_recursive(child) for child in node.children.values()]
                # Replace the maximum subtree token with its corresponding token and its children
                tokenized_tree = [max_subtree_token] + [t for child_tokens in tokenized_children for t in child_tokens]
                return tokenized_tree

        # Call tokenize_recursive on the input tree
        tokenized_tree = tokenize_recursive(tree)
        # Get unique trees by their string representation
        unique_trees = list(set([str(t) for t in tokenized_tree]))
        # Map the unique string representations back to the original trees
        result_trees = [next(t for t in tokenized_tree if str(t) == ut) for ut in unique_trees]

        return result_trees

    # def tokenize(self, tree):
    #     def tokenize_recursive(node):
    #         max_subtree_size = 0
    #         max_subtree_token = None
    #
    #         for token in self.tokens:
    #             if token.is_nonterminal:
    #                 subtree_size = token.count_nodes()
    #                 matches = self._find_subtree(node, token)
    #                 if len(matches) > 0 and subtree_size > max_subtree_size:
    #                     max_subtree_size = subtree_size
    #                     max_subtree_token = token
    #
    #         if max_subtree_token is None:
    #             tokenized_tree = [node]
    #             if not node.is_terminal:
    #                 for child in node.children.values():
    #                     tokenized_tree.extend(tokenize_recursive(child))
    #             return tokenized_tree
    #         else:
    #             return [max_subtree_token]
    #
    #     tokenized_tree = tokenize_recursive(tree)
    #     unique_trees = list(set([str(t) for t in tokenized_tree]))
    #     result_trees = [next(t for t in tokenized_tree if str(t) == ut) for ut in unique_trees]
    #     return result_trees
