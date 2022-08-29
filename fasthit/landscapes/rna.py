"""Defines RNA binding landscape and problem registry."""
from typing import Dict, Sequence

import numpy as np
# ViennaRNA is an optional dependency
try:
    import RNA
except ImportError:
    pass

import fasthit


class RNAFolding(fasthit.Landscape):
    """RNA folding landscape using ViennaRNA `fold`."""

    def __init__(self, norm_value=1):
        """Create an RNAFolding landscape."""
        super().__init__(name="RNAFolding")

        self._sequences = {}
        self._norm_value = norm_value

    def _fitness_function(self, sequence: Sequence[str]) -> np.ndarray:
        _, fe = RNA.fold(sequence)
        return -fe.astype(np.float32) / self._norm_value


class RNABinding(fasthit.Landscape):
    """RNA binding landscape using ViennaRNA `duplexfold`."""

    def __init__(
        self,
        targets: Sequence[str],
        seq_length: int,
        conserved_region: Dict = None,
    ):
        """
        Create an RNABinding landscape.

        Args:
            targets: List of RNA strings that will be binding targets.
                If more than one, the fitness score is the mean of each binding fitness.
            seq_length: Length of sequences to be evaluated.
            conserved_region: A dictionary of the form `{start: int, pattern: str}`
                defining the start of the conserved region and the pattern that must be
                conserved. Sequences violating these criteria will receive a score of
                zero (useful for creating `swampland` areas).

        """
        # ViennaRNA is not available through pip, so give a warning message
        # if not installed.
        try:
            RNA
        except NameError as e:
            raise ImportError(
                f"{e}.\n"
                "Hint: ViennaRNA not installed.\n"
                "      Source and binary installations available at "
                "https://www.tbi.univie.ac.at/RNA/#download.\n"
                "      Conda installation available at "
                "https://anaconda.org/bioconda/viennarna."
            ) from e
        super().__init__(name=f"RNABinding_T{targets}_L{seq_length}")

        self._targets = targets
        self._seq_length = seq_length
        self._conserved_region = conserved_region
        self._norm_values = self._compute_min_binding_energies()

        self._sequences = {}

    def _fitness_function(self, sequences: Sequence[str]) -> np.ndarray:
        fitnesses = []
        for seq in sequences:
            # Check that sequence is of correct length
            if len(seq) != self._seq_length:
                raise ValueError(
                    f"All sequences in `sequences` must be of length {self._seq_length}"
                )
            # If `self._conserved_region` is not None, check that the region is conserved
            if self._conserved_region is not None:
                start = self._conserved_region["start"]
                pattern = self._conserved_region["pattern"]
                # If region not conserved, fitness is 0
                if seq[start: start + len(pattern)] != pattern:
                    fitnesses.append(0)
                    continue
            # Energy is sum of binding energies across all targets
            energies = np.array(
                [RNA.duplexfold(
                    target, seq).energy for target in self._targets]
            )
            fitness = (energies / self._norm_values).mean()
            fitnesses.append(fitness)
        return np.array(fitnesses, dtype=np.float32)

    def _compute_min_binding_energies(self):
        """Compute the lowest possible binding energy for each target."""
        complements = {"A": "U", "C": "G", "G": "C", "U": "A"}

        min_energies = []
        for target in self._targets:
            complement = "".join(complements[x] for x in target)[::-1]
            energy = RNA.duplexfold(complement, target).energy
            min_energies.append(energy * self._seq_length / len(target))

        return np.array(min_energies)


def registry():
    """
    Return a dictionary of problems of the form:
    `{
        "problem name": {
            "params": ...,
            "starts": ...
        },
        ...
    }`

    where `fasthit.landscapes.RNABinding(**problem["params"])` instantiates the
    RNA binding landscape for the given set of parameters.

    Returns:
        dict: Problems in the registry.

    """
    # RNA target sequences
    targets = [
        "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACCCCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",  # noqa: E501
        "GAGGCACAUUCCGGCUCGCCCCCGUCCGCGCGGGGGCCCCGCGCGGACGGGGUCCGGCCCGCGCGGGGCCCCCGCGCGGGAGCCGGAAUGUGCCUCGUUC",  # noqa: E501
        "CCGGUGAUACUGUUAGUGGUCACGGUGCAUUUAUAGCGCUAAAGUACAGUCUUCCCCUGUUGAACGGCGCCAUUGCAUACAGGGCCAGCCGCGUAACGCC",  # noqa: E501
        "UAAGAGAGCGUAAAAAUAGAGAUAUGUUCUUGGGUCAGGGCUAUGCGUACCCCAUGAGAGUAAAUCAUACCCCCAAUGGGCUUCGGCGGAAAUUCACUUA",  # noqa: E501
    ]

    # Starting sequences of lengths 14, 50, and 100
    starts = {
        14: [
            "AUGGGCCGGACCCC",
            "GCCCCGCCGGAAUG",
            "UCUUGGGGACUUUU",
            "GGAUAACAAUUCAU",
            "CCCAUGCGCGAUCA",
        ],
        50: [
            "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACC",
            "CCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",
            "AUGUUUCUUUUAUUUAUCUGAGCAUGGGCGGGGCAUUUGCCCAUGCAAUU",
            "UAAACGAUGCUUUUGCGCCUGCAUGUGGGUUAGCCGAGUAUCAUGGCAAU",
            "AGGGAAGAUUAGAUUACUCUUAUAUGACGUAGGAGAGAGUGCGGUUAAGA",
        ],
        100: [
            "GAACGAGGCACAUUCCGGCUCGCCCGGCCCAUGUGAGCAUGGGCCGGACCCCGUCCGCGCGGGGCCCCCGCGCGGACGGGGGCGAGCCGGAAUGUGCCUC",  # noqa: E501
            "AGCAUCUCGCCGUGGGGGCGGGCCCGGCCCAUGUGAGCAUGCGUAGGUUUAUCCCAUAGAGGACCCCGGGAGAACUGUCCAAUUGGCUCCUAGCCCACGC",  # noqa: E501
            "GGCGGAUACUAGACCCUAUUGGCCCGGCCCAUGUGAGCAUGGCCCCAGAUCUUCCGCUCACUCGCAUAUUCCCUCCGGUUAAGUUGCCGUUUAUGAAGAU",  # noqa: E501
            "UUGCAGGUCCCUACACCUCCGGCCCGGCCCAUGUGACCAUGAAUAGUCCACAUAAAAACCGUGAUGGCCAGUGCAGUUGAUUCCGUGCUCUGUACCCUUU",  # noqa: E501
            "UGGCGAUGAGCCGAGCCGCCAUCGGACCAUGUGCAAUGUAGCCGUUCGUAGCCAUUAGGUGAUACCACAGAGUCUUAUGCGGUUUCACGUUGAGAUUGCA",  # noqa: E501
        ],
    }

    problems = {}

    # Single target problems - 4 of these
    for t in range(len(targets)):
        for length, start in starts.items():
            name = f"L{length}_RNA{t+1}"
            problems[name] = {
                "params": {"targets": [targets[t]], "seq_length": length},
                "starts": start,
            }

    # Two-target problems
    for t1 in range(len(targets)):
        for t2 in range(t1 + 1, len(targets)):
            for length, start in starts.items():
                name = f"L{length}_RNA{t1+1}+{t2+1}"
                problems[name] = {
                    "params": {
                        "targets": [targets[t1], targets[t2]],
                        "seq_length": length,
                    },
                    "starts": start,
                }

    # Two-target problems with conserved portion
    for t1 in range(len(targets)):
        for t2 in range(t1 + 1, len(targets)):
            name = f"C20_L100_RNA{t1+1}+{t2+1}"
            problems[name] = {
                "params": {
                    "targets": [targets[t1], targets[t2]],
                    "seq_length": 100,
                    "conserved_region": {
                        "start": 21,
                        "pattern": "GCCCGGCCCAUGUGAGCAUG",
                    },
                },
                "starts": starts[100],
            }

    return problems
