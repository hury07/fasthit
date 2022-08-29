import math
import os
import shutil
import string
import subprocess
from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from fasthit.encoders.esm import ESM, Pretrained
from fasthit.error import EsmImportError
from joblib import Parallel, delayed

try:
    import esm
except ImportError:
    pass


_homedir = os.path.expanduser("~")

encodings = pd.DataFrame(
    {
        "encoder": [
            "esm-msa-1", "esm-msa-1b"
        ],
        "model": [
            "esm_msa1_t12_100M_UR50S", "esm_msa1b_t12_100M_UR50S"
        ],
        "n_features": [768, 768],
    }
)
encodings.set_index("encoder", inplace=True)


class ESMMsa(ESM):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
        pretrained_model_dir: Optional[str] = None,
        database: Optional[str] = None,
        msa_depth: int = 64,
        msa_batch_size:  int = 8,
        n_threads: int = 8,
    ):
        try:
            esm
        except Exception:
            raise EsmImportError

        assert encoding in ["esm-msa-1", "esm-msa-1b"]
        name = f"ESM_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_list = list(wt_seq)
        self._target_python_idxs = target_python_idxs
        self._embeddings: Dict[str, np.ndarray] = {}

        self._database = _homedir + \
            "/databases/hhsuite/uniclust30/UniRef30_2020_06" if database is None else database
        self._msa_depth = msa_depth
        self._msa_batch_size = msa_batch_size
        self._n_threads = n_threads

        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')

        self._load_model(target_python_idxs, pretrained_model_dir)

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size,
        )

    def _load_model(self, target_python_idxs: Sequence[int], pretrained_model_dir: str):
        if pretrained_model_dir is None:
            pretrained_model_dir = _homedir + "/databases/pretrained_models/esm/"
        pretrained_model_file = pretrained_model_dir + \
            self._encoding["model"]+".pt"
        if not os.path.isfile(pretrained_model_file):
            pretrained_model_file = self._encoding["model"]
        pretrained_model, esm_alphabet = Pretrained.load_model_and_alphabet(
            pretrained_model_file)
        ###
        self._pretrained_model = pretrained_model.eval().to(self._device)
        self._batch_converter = esm_alphabet.get_batch_converter()
        self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]

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

    @staticmethod
    def _remove_insertions(sequence: str) -> str:
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        return sequence.translate(translation)

    def _filt_msa(self, file: str, msa_depth: int, temp_dir: str):
        labels = []
        seqs = []
        for record in SeqIO.parse(f"{temp_dir}/{file}.a3m", "fasta"):
            labels.append(record.id)
            seqs.append(self._remove_insertions(str(record.seq)))
        ###
        seq_ref = seqs[0]
        label_ref = labels[0]
        seqs = seqs[1:]
        labels = labels[1:]
        dist = []
        for seq in seqs:
            dist.append(self._hamming_distance(seq, seq_ref))
        ###
        dist = np.array(dist)
        msa_depth = min(len(dist), msa_depth)
        sort_order = np.argsort(dist)[:-msa_depth-1:-1]
        return [(label_ref, seq_ref)] + [(labels[idx], seqs[idx]) for idx in sort_order]

    def _get_msa(self, seq: str, name: str):
        with open(f"{self.temp_dir}/{name}.fasta", "w") as f:
            SeqIO.write(SeqRecord(Seq(seq), id=name,
                        description=""), f, "fasta")
        # run hhblits
        _ = subprocess.run(
            [
                "hhblits", "-i", f"{self._temp_dir}/{name}.fasta",
                "-oa3m", f"{self._temp_dir}/{name}.a3m",
                "-d", self._database, "-v", "1", "-cpu", str(self._n_threads),
                "-diff", str(self._msa_depth)
            ]
        )
        ###
        # filter msa
        labels = []
        seqs = []
        for record in SeqIO.parse(f"{self._temp_dir}/{name}.a3m", "fasta"):
            labels.append(record.id)
            seqs.append(self._remove_insertions(str(record.seq)))
        ###
        seq_ref = seqs[0]
        label_ref = labels[0]
        seqs = seqs[1:]
        labels = labels[1:]
        dist = [self._hamming_distance(seq, seq_ref) for seq in seqs]

        dist = np.array(dist)
        msa_depth = min(len(dist), msa_depth)
        sort_order = np.argsort(dist)[:-msa_depth-1:-1]
        return [(label_ref, seq_ref)] + [(labels[idx], seqs[idx]) for idx in sort_order]
    
    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        return self.encode_func(sequences, self._embed_msa, None)
    
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
