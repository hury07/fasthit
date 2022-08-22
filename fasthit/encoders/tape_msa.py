import math
import os
import shutil
import subprocess
from typing import Dict, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from fasthit.encoders.tape import TAPE
from fasthit.error import TapeImportError
from joblib import Parallel, delayed

_homedir = os.path.expanduser("~")

encodings = pd.DataFrame(
    {
        "encoder": ["trrosetta"],
        "model": ["xaa"],
        "n_features": [526],
    }
)
encodings.set_index("encoder", inplace=True)


class TAPEMsa(TAPE):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
        database: str = _homedir + "/databases/hhsuite/uniclust30/UniRef30_2020_06",
        msa_depth: int = 64,
        msa_batch_size:  int = 8,
        n_threads: int = 8,
    ):
        assert encoding in ["trrosetta"]
        name = f"TAPE_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_list = list(wt_seq)
        self._target_python_idxs = target_python_idxs
        self._embeddings: Dict[str, np.ndarray] = {}

        self._database = database
        self._msa_depth = msa_depth
        self._msa_batch_size = msa_batch_size
        self._n_threads = n_threads

        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')

        self._load_model(target_python_idxs)

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size
        )

    def _load_model(self, target_python_idxs: Sequence[int]):
        try:
            import tape
        except ImportError:
            raise TapeImportError

        pretrained_model = tape.TRRosetta.from_pretrained(
            self._encoding["model"])
        self._pretrained_model = pretrained_model.to(self._device)
        self._pretrained_model.eval()

        self._target_protein_idxs = [
            idx + 1 for idx in target_python_idxs]  # TODO check

        from datetime import datetime
        self._temp_dir = f".tmp.{datetime.now().timestamp()}"

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        return self.encode_func(sequences, self._embed_msa, None)

    @staticmethod
    def _hamming_distance(s1: str, s2: str):
        # Return the Hamming distance between equal-length sequences
        if len(s1) != len(s2):
            raise ValueError(
                "Undefined for sequences of unequal length")
        return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

    def _filt_msa(self, file: str, msa_depth: int, temp_dir: str):
        records = []
        with open(f"{temp_dir}/{file}.txt") as f:
            for line in f:
                line = line.strip("\n")
                records.append(line.split(" ")[-1])
        ###
        seq_ref = records[0]
        records = np.array(records[1:])
        dist = []
        for record in records:
            dist.append(self._hamming_distance(record, seq_ref))
        ###
        dist = np.array(dist)
        msa_depth = min(len(dist), msa_depth)
        sort_order = np.argsort(dist)[:-msa_depth-1:-1]
        return np.append(records[sort_order], seq_ref).tolist()

    def _get_msa(self, seq: str, name: str):
        with open(f"{self._temp_dir}/{name}.fasta", "w") as f:
            SeqIO.write(SeqRecord(Seq(seq), id=name,
                        description=""), f, "fasta")
        # run hhblits
        _ = subprocess.run(
            [
                "hhblits", "-i", f"{self._temp_dir}/{name}.fasta",
                "-opsi", f"{self._temp_dir}/{name}.txt",
                "-d", self._database, "-v", "1", "-cpu", str(self._n_threads),
                "-diff", str(self._msa_depth)
            ]
        )

        # filter msa
        records = []
        with open(f"{self._temp_dir}/{name}.txt") as f:
            for line in f:
                line = line.strip("\n")
                records.append(line.split(" ")[-1])

        seq_ref = records[0]
        seqs = np.array(records[1:])
        dist = [self._hamming_distance(seq, seq_ref) for seq in seqs]

        dist = np.array(dist)
        msa_depth = min(len(dist), msa_depth)
        sort_order = np.argsort(dist)[:-msa_depth-1:-1]
        msa = np.append(records[sort_order], seq_ref).tolist()

        label = [name for _ in range(len(msa))]
        return msa, label

    def _embed_msa(
        self,
        data: Sequence[Tuple[Sequence[str], Sequence[str]]],
    ) -> np.ndarray:
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)

        n_batches = math.ceil(len(data) / self._msa_batch_size)
        results = []
        with Parallel(n_jobs=n_batches) as parallel:
            for batch in np.array_split(data, n_batches):
                batch = cast(Tuple[Sequence[str], Sequence[str]], batch)
                msas = parallel(
                    delayed(self._get_msa)(name, seq) for name, seq in batch
                )
                results.append(self._embed(msas))
        shutil.rmtree(self._temp_dir)
        return np.concatenate(results, axis=0)
