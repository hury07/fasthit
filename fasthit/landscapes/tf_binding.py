"""Define TFBinding landscape and problem registry."""
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import fasthit


class TFBinding(fasthit.Landscape):
    """
    A landscape of binding affinity of proposed 8-mer DNA sequences to a
    particular transcription factor.

    We use experimental data from Barrera et al. (2016), a survey of the binding
    affinity of more than one hundred and fifty transcription factors (TF) to all
    possible DNA sequences of length 8.
    """

    def __init__(self, landscape_file: str):
        """
        Create a TFBinding landscape from experimental data .csv file.

        See https://github.com/samsinai/FLSD-Sandbox/tree/stewy-redesign/fasthit/landscapes/data/tf_binding  # noqa: E501
        for examples.
        """
        super().__init__(name="TF_Binding")

        # Load TF pairwise TF binding measurements from file
        data = pd.read_csv(landscape_file, sep="\t")
        score = data["E-score"]  # "E-score" is enrichment score
        norm_score = (score - score.min()) / (score.max() - score.min())

        # The csv file keeps one DNA strand's sequence in "8-mer" and the other in
        # "8-mer.1".
        # Since it doesn't really matter which strand we have, we will map the sequences
        # of both strands to the same normalized enrichment score.
        self.sequences = dict(zip(data["8-mer"], norm_score))
        self.sequences.update(zip(data["8-mer.1"], norm_score))

    def _fitness_function(self, sequences: List[str]) -> np.ndarray:
        return np.array([self.sequences[seq] for seq in sequences])


def registry() -> Dict[str, Dict]:
    """
    Return a dictionary of problems of the form:

    ```python
    {
        "problem name": {
            "params": ...,
        },
        ...
    }
    ```

    where `fasthit.landscapes.TFBinding(**problem["params"])` instantiates the
    transcription factor binding landscape for the given set of parameters.

    Returns:
        Problems in the registry.

    """
    tf_binding_data_dir = os.path.join(os.path.dirname(__file__), "data/tf_binding")

    problems = {}
    for fname in os.listdir(tf_binding_data_dir):
        problem_name = fname.replace("_8mers.txt", "")

        problems[problem_name] = {
            "params": {"landscape_file": os.path.join(tf_binding_data_dir, fname)},
            "starts": [
                "GCTCGAGC",
                "GCGCGCGC",
                "TGCGCGCC",
                "ATATAGCC",
                "GTTTGGTA",
                "ATTATGTT",
                "CAGTTTTT",
                "AAAAATTT",
                "AAAAACGC",
                "GTTGTTTT",
                "TGCTTTTT",
                "AAAGATAG",
                "CCTTCTTT",
                "AAAGAGAG",
            ],
        }

    return problems
