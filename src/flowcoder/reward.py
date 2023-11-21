# To measure edit distance between outputs
import Levenshtein

from flowcoder.config import *
from deepsynth.experiment_helper import *
from deepsynth.program import Program

from itertools import chain


def normalized_similarity(seq1, seq2):
    # Computes the normalized edit distance
    ld = 1 - Levenshtein.distance(seq1, seq2) / (max(len(seq1), len(seq2)) + 1e-10)
    return ld

def rewards(programs, batch_ios, dsl, lexicon, max_reward=1):
    # Create program checkers
    program_checkers = [make_program_checker(dsl, examples, lexicon, max_reward)
                        for examples in batch_ios]

    # Compute rewards
    reward = [float(program_checker(program)) for program, program_checker in zip(programs, program_checkers)]

    return torch.tensor(reward, requires_grad=True, device=device)


def make_program_checker(dsl: DSL, examples, data_lexicon, max_reward=1) -> Callable[[Program, bool], float]:
    def checker(prog: Program) -> float:
        predicted_output = [prog.eval_naive(dsl, example[0]) for example in examples]

        # Check additional conditions
        if prog.is_constant() or \
                predicted_output is None or \
                None in predicted_output or \
                None in chain.from_iterable(predicted_output) or \
                not all(out in data_lexicon for sublist in predicted_output for out in sublist):
            return 0.0

        # Compute the average normalized Levenshtein distance over all examples
        avg_levenshtein_distance = 0
        for example in examples:
            input, output = example
            out = prog.eval_naive(dsl, input)
            avg_levenshtein_distance += normalized_similarity(output, out)
        avg_levenshtein_distance /= len(examples)
        return max_reward if avg_levenshtein_distance == 1 else avg_levenshtein_distance

    return checker
