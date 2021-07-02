###
import os
from typing import Sequence

import math
import numpy as np
import pandas as pd
import torch

import fasthit

_filedir = os.path.dirname(os.path.abspath(__file__))
encodings = pd.DataFrame(
    {
        "encoder": ["esm-1b", "esm-msa-1"],
        "model": ["esm1b_t33_650M_UR50S", "esm_msa1_t12_100M_UR50S"],
        "n_features": [1280, 768],
    }
)
encodings.set_index("encoder", inplace=True)


class ESM(fasthit.Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
        pretrained_model_dir: str = _filedir + "/data/esm/",
        database: str = "/home/hury/databases/hhsuite/uniclust30/UniRef30_2020_06",
        msa_depth: int = 64,
        msa_batch_size:  int = 8,
        n_threads: int = 8,
    ):
        assert encoding in ["esm-1b", "esm-msa-1"]
        name = f"ESM_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_seq = wt_seq
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}
        
        self._device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
        from esm import pretrained
        self._pretrained_model, self._esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + self._encoding["model"]+".pt"
        )
        self._pretrained_model = self._pretrained_model.to(self._device)
        self._pretrained_model.eval()
        self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        ### For MSA-Transformer
        if self._encoding.name in ["esm-msa-1"]:
            self._database = database
            self._msa_depth = msa_depth
            self._msa_batch_size = msa_batch_size
            self._n_threads = n_threads

            from datetime import datetime
            self._temp_dir = f".tmp.{datetime.now().timestamp()}"
        
        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size,
        )
    
    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        ### [bsz, seq_length, n_features]
        encoded_idx = [
            idx for idx in range(len(sequences)) if sequences[idx] in self._embeddings.keys()
        ]
        encoded = [self._embeddings[sequences[idx]] for idx in encoded_idx]
        if len(encoded_idx) == len(sequences):
            return np.array(encoded, dtype=np.float32)
        ###
        unencoded_idx = [idx for idx in range(len(sequences)) if idx not in encoded_idx]
        unencoded_seqs = [sequences[idx] for idx in unencoded_idx]
        n_batches = math.ceil(len(unencoded_seqs) / self.batch_size)
        ###
        extracted_embeddings = [None for _ in range(n_batches)]
        wt_list = list(self._wt_seq)
        # Loop over the number of batches
        for i, combo_batch in enumerate(np.array_split(unencoded_seqs, n_batches)):
            temp_seqs = [None for _ in range(len(combo_batch))]
            for j, combo in enumerate(combo_batch):
                temp_seq = wt_list.copy()
                # Loop over the target python indices and set new amino acids
                for aa, ind in zip(combo, self._target_python_idxs):
                    temp_seq[ind] = aa
                temp_seqs[j] = "".join(temp_seq)
            if self._encoding.name in ["esm-1b"]:
                extracted_embeddings[i] = self._embed(temp_seqs, combo_batch)
            elif self._encoding.name in ["esm-msa-1"]:
                extracted_embeddings[i] = self._embed_msa(temp_seqs, combo_batch)
        unencoded = np.concatenate(extracted_embeddings, axis=0)
        embeddings = np.empty((len(sequences), *unencoded.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(encoded_idx):
            embeddings[idx] = encoded[i]
        for i, idx in enumerate(unencoded_idx):
            embeddings[idx] = unencoded[i]
        return embeddings
    
    def _embed(
        self,
        sequences: Sequence[str],
        labels: Sequence[str],
        toks_per_batch: int = 4096,
    ) -> np.ndarray:
        ###
        from esm import FastaBatchedDataset
        ###
        dataset = FastaBatchedDataset(labels, sequences)
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self._esm_alphabet.get_batch_converter(), batch_sampler=batches
        )
        ###
        results = []
        for labels, _, toks in data_loader:
            toks = toks.to(self._device)
            with torch.no_grad():
                out = self._pretrained_model(
                    toks, repr_layers=[self._pretrained_model.num_layers], return_contacts=False
                )
            embeddings = out["representations"][self._pretrained_model.num_layers]
            if embeddings.ndim == 4:
                # query_sequence at the last row of MSA
                embeddings = embeddings[:, -1]
                labels = labels[-1]
            embeddings = embeddings[:, self._target_protein_idxs].detach().cpu().numpy()
            repr_dict = {labels[idx]: embeddings[idx] for idx in range(embeddings.shape[0])}
            self._embeddings.update(repr_dict)
            results.append(embeddings)
        torch.cuda.empty_cache()
        return np.concatenate(results, axis=0)
    
    def _embed_msa(
        self,
        sequences: Sequence[str],
        seq_names: Sequence[str],
    ) -> np.ndarray:
        ###
        import shutil
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        import subprocess
        from joblib import Parallel, delayed
        ###
        def _get_msa(seq, name, **kwargs):
            ###
            def _hamming_distance(s1, s2):
                #Return the Hamming distance between equal-length sequences
                if len(s1) != len(s2):
                    raise ValueError("Undefined for sequences of unequal length")
                return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
            ###
            def _filt_msa(file, msa_depth, temp_dir):
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
                    dist.append(_hamming_distance(record, seq_ref))
                ###
                dist = np.array(dist)
                msa_depth = min(len(dist), msa_depth)
                sort_order = np.argsort(dist)[:-msa_depth-1:-1]
                return np.append(records[sort_order], seq_ref).tolist()
            ###
            temp_dir = kwargs["temp_dir"]
            with open(f"{temp_dir}/{name}.fasta", "w") as f:
                SeqIO.write(SeqRecord(Seq(seq), id = name,  description=""), f, "fasta")
            ###
            _ = subprocess.run(
                [
                    "hhblits", "-i", f"{temp_dir}/{name}.fasta", "-opsi", f"{temp_dir}/{name}.txt",
                    "-d", kwargs["database"], "-v", "1", "-cpu", str(kwargs["n_threads"]),
                    "-diff", str(kwargs["msa_depth"])
                ]
            )
            ###
            msa = _filt_msa(name, kwargs["msa_depth"], temp_dir)
            label = [name for _ in range(len(msa))]
            return msa, label
        ###
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)
        kwargs = {
            "database": self._database,
            "msa_depth": self._msa_depth,
            "n_threads": self._n_threads,
            "temp_dir": self._temp_dir,
        }
        n_batches = math.ceil(len(sequences) / self._msa_batch_size)
        results = []
        with Parallel(n_jobs=self._msa_batch_size) as parallel:
            for seqs_batch, names_batch in zip(
                np.array_split(sequences, n_batches),
                np.array_split(seq_names, n_batches)
            ):
                msas_and_labels = parallel(
                    delayed(_get_msa)(seq, name, **kwargs) for seq, name in zip(seqs_batch, names_batch)
                )
                for msa_and_label in msas_and_labels:
                    results.append(self._embed(*msa_and_label))
        shutil.rmtree(self._temp_dir)
        return np.concatenate(results, axis=0)