###

from fasthit.utils.ffindex_util import ffindex_to_record
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
        "encoder": [
            "esm-1b", "esm-1v",
            "esm-msa-1", "esm-msa-1b"
            ],
        "model": [
            "esm1b_t33_650M_UR50S", "esm1v_t33_650M_UR90S_1",
            "esm_msa1_t12_100M_UR50S", "esm_msa1b_t12_100M_UR50S"
            ],
        "n_features": [1280, 1280, 768, 768],
    }
)
encodings.set_index("encoder", inplace=True)


class ESMMPI(fasthit.Encoder):
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
        maxfilt: int = 50000,
        msa_batch_size:  int = 8,
        n_threads: int = 4,
        mpi_cpus: int = 32
    ):
        assert encoding in ["esm-1b", "esm-msa-1"]
        name = f"ESM_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_seq = wt_seq
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}

        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
        from esm import pretrained
        self._pretrained_model, self._esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + self._encoding["model"]+".pt"
        )
        self._pretrained_model = self._pretrained_model.to(self._device)
        self._pretrained_model.eval()
        self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        # For MSA-Transformer
        if self._encoding.name in ["esm-msa-1"]:
            self._database = database
            self._msa_depth = msa_depth
            self._maxfilt = maxfilt
            self._msa_batch_size = msa_batch_size
            self._n_threads = n_threads
            self._mpi_cpus = mpi_cpus

            from datetime import datetime
            self._temp_dir = f".tmp.{datetime.now().timestamp()}"

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size,
        )

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        # [bsz, seq_length, n_features]
        encoded_idx = [
            idx for idx in range(len(sequences)) if sequences[idx] in self._embeddings.keys()
        ]
        encoded = [self._embeddings[sequences[idx]] for idx in encoded_idx]
        if len(encoded_idx) == len(sequences):
            return np.array(encoded, dtype=np.float32)
        ###
        unencoded_idx = [idx for idx in range(
            len(sequences)) if idx not in encoded_idx]
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
                extracted_embeddings[i] = self._embed_msa(
                    temp_seqs, combo_batch)
        unencoded = np.concatenate(extracted_embeddings, axis=0)
        embeddings = np.empty(
            (len(sequences), *unencoded.shape[1:]), dtype=np.float32)
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
        batches = dataset.get_batch_indices(
            toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self._esm_alphabet.get_batch_converter(), batch_sampler=batches
        )
        ###
        results = []
        for labels, _, toks in data_loader:
            toks = toks.to(self._device)
            with torch.no_grad():
                out = self._pretrained_model(
                    toks, repr_layers=[
                        self._pretrained_model.num_layers], return_contacts=False
                )
            embeddings = out["representations"][self._pretrained_model.num_layers]
            embedding = embeddings.detach().cpu().numpy()
            del toks, embeddings
            torch.cuda.empty_cache()
            ###
            if embedding.ndim == 4:
                # query_sequence at the last row of MSA
                embedding = embedding[:, -1]
                labels = labels[-1]
            embedding = embedding[:, self._target_protein_idxs]
            results.append(embedding)
            repr_dict = {labels[idx]: embedding[idx]
                         for idx in range(embedding.shape[0])}
            self._embeddings.update(repr_dict)
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

        def _write_msa_mpi(seq, name, name_index, name_used, **kwargs):
            temp_dir = kwargs["temp_dir"]
            if name_used[name_index[name]] == True:
                return
            name_used[name_index[name]] = True
            with open(f"{temp_dir}/{name}.fasta", "w") as f:
                SeqIO.write(SeqRecord(Seq(seq), id=name,
                            description=""), f, "fasta")
            _ = subprocess.run(
                [
                    "ffindex_from_fasta", "-s",
                    f"{temp_dir}/{name}.ffdata",
                    f"{temp_dir}/{name}.ffindex",
                    f"{temp_dir}/{name}.fasta"
                ]
            )

        def _call_hhblits(**kwargs):
            temp_dir = kwargs["temp_dir"]

            names = set(f[:-len(".fasta")]
                        for f in os.listdir(temp_dir) if f.endswith('fasta'))

            # merge seqs to all
            for i, name in enumerate(names):
                if i == 0:
                    build_cmd = ["ffindex_build",
                                 f"{temp_dir}/seq_sum.ffdata", f"{temp_dir}/seq_sum.ffindex",
                                 "-d",  f"{temp_dir}/{name}.ffdata", "-i", f"{temp_dir}/{name}.ffindex"]
                elif i == len(names) - 1:
                    build_cmd = ["ffindex_build", "-sa",
                                 f"{temp_dir}/seq_sum.ffdata", f"{temp_dir}/seq_sum.ffindex",
                                 "-d",  f"{temp_dir}/{name}.ffdata", "-i", f"{temp_dir}/{name}.ffindex"]
                else:
                    build_cmd = ["ffindex_build", "-a",
                                 f"{temp_dir}/seq_sum.ffdata", f"{temp_dir}/seq_sum.ffindex",
                                 "-d",  f"{temp_dir}/{name}.ffdata", "-i", f"{temp_dir}/{name}.ffindex"]
                subprocess.run(build_cmd)

            # call hhblits
            _ = subprocess.run(
                [
                    "mpirun", "-c", str(kwargs["mpi_cpus"]),
                    "hhblits_mpi", "-i", f"{temp_dir}/seq_sum", "-opsi", f"{temp_dir}/search_result",
                    "-d", kwargs["database"], "-v", "1", "-maxfilt", str(kwargs["maxfilt"]), "-maxres", str(kwargs["maxfilt"]+1), 
                    "-cpu", str(kwargs["mpi_cpus"]), "-diff", str(kwargs["msa_depth"])
                ]
            )

            # get record from ffindex
            db_records = ffindex_to_record(f"{temp_dir}/search_result")
            return db_records

        def _get_msa_mpi(name, records, **kwargs):
            ###
            def _hamming_distance(s1, s2): return sum(
                ch1 != ch2 for ch1, ch2 in zip(s1, s2))
            ###
            msa_depth = kwargs["msa_depth"]

            ###
            seq_ref = records[0]
            records = np.array(records[1:])
            dist = [_hamming_distance(record, seq_ref) for record in records]
            ###
            dist = np.array(dist)
            msa_depth = min(len(dist), msa_depth)
            sort_order = np.argsort(dist)[:-msa_depth-1:-1]
            msa = np.append(records[sort_order], seq_ref).tolist()
            label = [name for _ in range(len(msa))]
            return msa, label

        def _write_msa(seq, name, name_index, name_used, **kwargs):
            temp_dir = kwargs["temp_dir"]
            if name_used[name_index[name]] == True:
                return
            name_used[name_index[name]] = True
            with open(f"{temp_dir}/{name}.fasta", "w") as f:
                SeqIO.write(SeqRecord(Seq(seq), id=name,
                            description=""), f, "fasta")

            _ = subprocess.run(
                [
                    "hhblits", "-i", f"{temp_dir}/{name}.fasta", "-opsi", f"{temp_dir}/{name}.txt",
                    "-d", kwargs["database"], "-v", "1", "-maxfilt", str(kwargs["maxfilt"]), "-maxres", str(kwargs["maxfilt"]+1), 
                    "-cpu", str(kwargs["n_threads"]), "-diff", str(kwargs["msa_depth"])
                ]
            )

        def _get_msa(name, **kwargs):
            ###
            def _hamming_distance(s1, s2): return sum(
                ch1 != ch2 for ch1, ch2 in zip(s1, s2))
            ###
            msa_depth = kwargs["msa_depth"]
            temp_dir = kwargs["temp_dir"]

            with open(f"{temp_dir}/{name}.txt") as f:
                lines = f.readlines()
            records = [l.rstrip('\n').split(" ")[-1] for l in lines]
            ###
            seq_ref = records[0]
            records = np.array(records[1:])
            dist = [_hamming_distance(record, seq_ref) for record in records]
            ###
            dist = np.array(dist)
            msa_depth = min(len(dist), msa_depth)
            sort_order = np.argsort(dist)[:-msa_depth-1:-1]
            msa = np.append(records[sort_order], seq_ref).tolist()
            label = [name for _ in range(len(msa))]
            return msa, label
        
        def whether_use_mpi(seq_names, sequences, msa_batch_size, **kwargs):
            temp_dir = kwargs["temp_dir"]
            dif_names = set(seq_names)
            name_used = np.memmap(f'{temp_dir}/name_used_mem', dtype=np.bool,
                   shape=len(dif_names), mode='w+')
            
            name_index = {name:i for i, name in enumerate(dif_names)}
            if len(seq_names) > 3:
                [_write_msa_mpi(seq, name, name_index, name_used, **kwargs) for name, seq in zip(seq_names, sequences)]
                db_records = _call_hhblits(**kwargs)
                msas_and_labels = [_get_msa_mpi(name, db_records[name], **kwargs) for name in seq_names]
            else:
                n_batches = math.ceil(len(sequences) / msa_batch_size)
                msas_and_labels = []
                with Parallel(n_jobs=msa_batch_size, require='sharedmem') as parallel:
                    for seqs_batch, names_batch in zip(
                        np.array_split(sequences, n_batches),
                        np.array_split(seq_names, n_batches)
                    ):
                        parallel(
                            delayed(_write_msa)(seq, name, name_index, name_used, **kwargs) for seq, name in zip(seqs_batch, names_batch)
                        )
                        one_msas_and_labels = parallel(
                            delayed(_get_msa)(name, **kwargs) for name in names_batch
                        )
                        msas_and_labels.extend(one_msas_and_labels)
            return msas_and_labels
        ###
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)
        kwargs = {
            "database": self._database,
            "msa_depth": self._msa_depth,
            "maxfilt": self._maxfilt,
            "n_threads": self._n_threads,
            "mpi_cpus": self._mpi_cpus,
            "temp_dir": self._temp_dir,
        }

        from datetime import datetime
        a = datetime.now()
        msa_batch_size = self._msa_batch_size
        msas_and_labels = whether_use_mpi(seq_names, sequences, msa_batch_size, **kwargs)
        results = [self._embed(*msa_and_label)
                   for msa_and_label in msas_and_labels]

        b = datetime.now()
        print('all', b-a)
        shutil.rmtree(self._temp_dir)
        return np.concatenate(results, axis=0)
