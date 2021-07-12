"""Interface between algorithm and measurements"""
import os
import time
from Bio import SeqIO
import numpy as np
import pandas as pd
from typing import Dict, Sequence

import fasthit

class EXP(fasthit.Landscape):
    def __init__(
        self,
        fitness_csv: str,
        wt_fasta: str,
        search_space: str = None,
        sequence_csv: str = None,
    ):
        super().__init__(name="Measurements")

        assert fitness_csv.endswith(".csv")
        assert wt_fasta.endswith(".fasta") or wt_fasta.endswith(".fa")
        if sequence_csv is not None:
            assert sequence_csv.endswith(".csv")
        else:
            sequence_csv = "outputs/proposed_seqs.csv"

        self._sequence_file = sequence_csv
        dir_path, _ = os.path.split(self._sequence_file)
        os.makedirs(dir_path, exist_ok=True)

        self._fitness_file = fitness_csv
        self._sequences = {}
        self._last_time = None
        
        self._wt = str(SeqIO.read(wt_fasta, format="fasta").seq)
        if search_space is not None:
            combo = search_space.split(",")
            self._combo_protein_idxs = [int(idxs[1:]) for idxs in combo]
            self._combo_python_idxs = [idxs - 1 for idxs in self._combo_protein_idxs]
            temp_seq = [idxs[0] for idxs in combo]
            assert all([
                self._wt[self._combo_python_idxs[i]] == temp_seq[i] for i in range(len(temp_seq))
            ])
        else:
            self._combo_protein_idxs = range(1, len(self._wt) + 1)
            self._combo_python_idxs = range(len(self._wt))

    def _write_sequences(self, sequences: Sequence[str]):
        seqs = pd.DataFrame({"Variants": sequences})
        with open(self._sequence_file, "w") as f:
            seqs.to_csv(f, index=False)

    def _read_fitness(self):
        while(True):
            if (self._last_time is None
                or os.path.getmtime(self._fitness_file) > self._last_time
            ):
                self._last_time = os.path.getmtime(self._fitness_file)
                ###
                data = pd.read_csv(self._fitness_file)
                measured_seqs = set(data["Variants"])
                interset_seqs = measured_seqs.intersection(self._proposed_seqs)
                if len(interset_seqs) == len(self._proposed_seqs):
                    break
                else:
                    print((
                        f"There exist proposed sequences that are not measured "
                        + f"or not given in fitness file: \"{self._fitness_file}\"."
                    ))
            time.sleep(1)
        self._sequences.update(zip(data["Variants"], data["Fitness"]))

    def _fitness_function(self, sequences: Sequence[str]) -> np.ndarray:
        self._proposed_seqs = set(sequences)
        self._write_sequences(sequences)
        self._read_fitness()
        return np.array(
            [self._sequences[seq] for seq in sequences],
            dtype=np.float32
        )
    
    @property
    def wt(self):
        return self._wt
    
    @property
    def combo_protein_idxs(self):
        return self._combo_protein_idxs
    
    @property
    def combo_python_idxs(self):
        return self._combo_python_idxs
    
    @property
    def explored_space_size(self):
        return len(self._sequences)
