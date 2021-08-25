"""Utility functions for manipulating sequences."""
from typing import List, Sequence, Union

import numpy as np
import torch

AAS = "ILVAGMFYWEDQNHCRKSTP"
"""str: Amino acid alphabet for proteins (length 20 - no stop codon)."""

RNAA = "UGCA"
"""str: RNA alphabet (4 base pairs)."""

DNAA = "TGCA"
"""str: DNA alphabet (4 base pairs)."""

BA = "01"
"""str: Binary alphabet '01'."""

BLOSUM62 = torch.Tensor([[3.9979, 1.6944, 2.4175, 0.6325, 0.275, 1.4777, 0.9458, 0.6304, 0.4089, 0.3305,
                          0.339, 0.3829, 0.3279, 0.3263, 0.6535, 0.3548, 0.3964, 0.4432, 0.7798, 0.3847, ],
                         [1.6944, 3.7966, 1.3142, 0.6019, 0.2845, 1.9943, 1.1546, 0.6921, 0.568, 0.3729,
                          0.2866, 0.4773, 0.31, 0.3807, 0.6423, 0.4739, 0.4283, 0.4289, 0.6603, 0.3711, ],
                         [2.4175, 1.3142, 3.6922, 0.9365, 0.337, 1.2689, 0.7451, 0.658, 0.3745, 0.4289,
                          0.3365, 0.4668, 0.369, 0.3394, 0.7558, 0.4201, 0.4565, 0.5652, 0.9809, 0.4431, ],
                         [0.6325, 0.6019, 0.9365, 3.9029, 1.0569, 0.7232, 0.4649, 0.5426, 0.4165, 0.7413,
                          0.5446, 0.7568, 0.5883, 0.5694, 0.868, 0.6127, 0.7754, 1.4721, 0.9844, 0.7541, ],
                         [0.275, 0.2845, 0.337, 1.0569, 6.8763, 0.3955, 0.3406, 0.3487, 0.4217, 0.4813,
                          0.6343, 0.5386, 0.8637, 0.493, 0.4204, 0.45, 0.5889, 0.9036, 0.5793, 0.4774, ],
                         [1.4777, 1.9943, 1.2689, 0.7232, 0.3955, 6.4815, 1.0044, 0.7084, 0.6103, 0.5003,
                          0.3465, 0.8643, 0.4745, 0.5841, 0.6114, 0.6226, 0.6253, 0.5986, 0.7938, 0.4239, ],
                         [0.9458, 1.1546, 0.7451, 0.4649, 0.3406, 1.0044, 8.1288, 2.7694, 1.3744, 0.3307,
                          0.299, 0.334, 0.3543, 0.652, 0.439, 0.3807, 0.344, 0.44, 0.4817, 0.2874, ],
                         [0.6304, 0.6921, 0.658, 0.5426, 0.3487, 0.7084, 2.7694, 9.8322, 2.1098, 0.4965,
                          0.3457, 0.6111, 0.486, 1.7979, 0.4342, 0.556, 0.5322, 0.5575, 0.5732, 0.3635, ],
                         [0.4089, 0.568, 0.3745, 0.4165, 0.4217, 0.6103, 1.3744, 2.1098, 38.1078, 0.3743,
                          0.2321, 0.5094, 0.2778, 0.4441, 0.45, 0.3951, 0.3589, 0.3853, 0.4309, 0.2818, ],
                         [0.3305, 0.3729, 0.4289, 0.7413, 0.4813, 0.5003, 0.3307, 0.4965, 0.3743, 5.4695,
                          1.6878, 1.9017, 0.9113, 0.96, 0.2859, 0.9608, 1.3083, 0.9504, 0.7414, 0.6792, ],
                         [0.339, 0.2866, 0.3365, 0.5446, 0.6343, 0.3465, 0.299, 0.3457, 0.2321, 1.6878,
                          7.3979, 0.8971, 1.5539, 0.6786, 0.3015, 0.5732, 0.7841, 0.9135, 0.6948, 0.5987, ],
                         [0.3829, 0.4773, 0.4668, 0.7568, 0.5386, 0.8643, 0.334, 0.6111, 0.5094, 1.9017,
                          0.8971, 6.2444, 1.0006, 1.168, 0.3658, 1.4058, 1.5543, 0.9656, 0.7913, 0.6413, ],
                         [0.3279, 0.31, 0.369, 0.5883, 0.8637, 0.4745, 0.3543, 0.486, 0.2778, 0.9113,
                          1.5539, 1.0006, 7.0941, 1.222, 0.3978, 0.8586, 0.9398, 1.2315, 0.9842, 0.4999, ],
                         [0.3263, 0.3807, 0.3394, 0.5694, 0.493, 0.5841, 0.652, 1.7979, 0.4441, 0.96,
                          0.6786, 1.168, 1.222, 13.506, 0.355, 0.917, 0.7789, 0.7367, 0.5575, 0.4729, ],
                         [0.6535, 0.6423, 0.7558, 0.868, 0.4204, 0.6114, 0.439, 0.4342, 0.45, 0.2859,
                          0.3015, 0.3658, 0.3978, 0.355, 19.5766, 0.3089, 0.3491, 0.7384, 0.7406, 0.3796, ],
                         [0.3548, 0.4739, 0.4201, 0.6127, 0.45, 0.6226, 0.3807, 0.556, 0.3951, 0.9608,
                          0.5732, 1.4058, 0.8586, 0.917, 0.3089, 6.6656, 2.0768, 0.7672, 0.6778, 0.4815, ],
                         [0.3964, 0.4283, 0.4565, 0.7754, 0.5889, 0.6253, 0.344, 0.5322, 0.3589, 1.3083,
                          0.7841, 1.5543, 0.9398, 0.7789, 0.3491, 2.0768, 4.7643, 0.9319, 0.7929, 0.7038, ],
                         [0.4432, 0.4289, 0.5652, 1.4721, 0.9036, 0.5986, 0.44, 0.5575, 0.3853, 0.9504,
                          0.9135, 0.9656, 1.2315, 0.7367, 0.7384, 0.7672, 0.9319, 3.8428, 1.6139, 0.7555, ],
                         [0.7798, 0.6603, 0.9809, 0.9844, 0.5793, 0.7938, 0.4817, 0.5732, 0.4309, 0.7414,
                          0.6948, 0.7913, 0.9842, 0.5575, 0.7406, 0.6778, 0.7929, 1.6139, 4.8321, 0.6889, ],
                         [0.3847, 0.3711, 0.4431, 0.7541, 0.4774, 0.4239, 0.2874, 0.3635, 0.2818, 0.6792,
                          0.5987, 0.6413, 0.4999, 0.4729, 0.3796, 0.4815, 0.7038, 0.7555, 0.6889, 12.8375, ]])
"""np.ndarray: fit for sequence of esm"""

def construct_mutant_from_sample(
    pwm_sample: np.ndarray, one_hot_base: np.ndarray
) -> np.ndarray:
    """Return one hot mutant, a utility function for some explorers."""
    one_hot = np.zeros(one_hot_base.shape)
    one_hot += one_hot_base
    i, j = np.nonzero(pwm_sample)  # this can be problematic for non-positive fitnesses
    one_hot[i, :] = 0
    one_hot[i, j] = 1
    return one_hot


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    """
    Return the one-hot representation of a sequence string according to an alphabet.

    Args:
        sequence: Sequence string to convert to one_hot representation.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        One-hot numpy array of shape `(len(sequence), len(alphabet))`.

    """
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out


def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])


def generate_single_mutants(wt: str, alphabet: str) -> List[str]:
    """Generate all single mutants of `wt`."""
    sequences = [wt]
    for i in range(len(wt)):
        tmp = list(wt)
        for j in range(len(alphabet)):
            tmp[i] = alphabet[j]
            sequences.append("".join(tmp))
    return sequences


def generate_random_sequences(
    length: int,
    number: int,
    alphabet: Sequence[str],
) -> List[str]:
    """Generate random sequences of particular length."""
    return [
        "".join([np.random.choice(alphabet) for _ in range(length)]) for _ in range(number)
    ]

def generate_random_mutant(
    sequence: str,
    mu: float,
    alphabet: Sequence[str],
) -> str:
    """
    Generate a mutant of `sequence` where each residue mutates with probability `mu`.

    So the expected value of the total number of mutations is `len(sequence) * mu`.

    Args:
        sequence: Sequence that will be mutated from.
        mu: Probability of mutation per residue.
        alphabet: Alphabet string.

    Returns:
        Mutant sequence string.

    """
    mutant = []
    for s in sequence:
        if np.random.rand() < mu:
            mutant.append(np.random.choice(alphabet))
        else:
            mutant.append(s)
    return "".join(mutant)
