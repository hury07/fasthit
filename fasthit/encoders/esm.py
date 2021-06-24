###
import os
from copy import deepcopy
from typing import Sequence

import math
import numpy as np
import pandas as pd
import torch

from esm import FastaBatchedDataset, pretrained
import fasthit

_filedir = os.path.dirname(os.path.abspath(__file__))

class ESM(fasthit.Encoder):
    encodings = pd.DataFrame(
        {
            "encoder": ["esm-1b", "esm-msa-1"],
            "model": ["esm1b_t33_650M_UR50S", "esm_msa1_t12_100M_UR50S"],
            "n_features": [1280, 768],
        }
    )
    encodings.set_index("encoder", inplace=True)

    def __init__(
        self,
        alphabet: str,
        encoding: str,
        wt_seq: str,
        target_protein_inds: Sequence[int],
        batch_size: int = 256,
        pretrained_model_dir: str = _filedir + "/data/esm/",
    ):
        assert encoding in ["esm-1b", "esm-msa-1"]
        name = f"esm_{encoding}"

        self.encoding = self.encodings.loc[encoding]
        self.wt_seq = wt_seq
        self.target_protein_inds = target_protein_inds
        self.n_positions_combined = len(target_protein_inds)
        
        self.pretrained_model, self.esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + self.encoding["model"]+".pt"
        )

        super().__init__(
            name,
            alphabet,
            self.encoding["n_features"],
            batch_size=batch_size,
        )
    
    def encode(self, sequences: Sequence[str]) -> np.array:
        ### [bsz, seq_length, n_features]
        """
        Encodes a given combinatorial space using tape.
        Unlike Georgiev and one-hot encodings, these encodings are context-
        aware. To save on system RAM, this task is split into n_batches number
        of batches.
        """
        # Build fasta files
        n_batches = math.ceil(len(sequences) / self.batch_size)
        # Create a list to store the names of the raw embedding files
        extracted_embeddings = [None for _ in range(n_batches)]
        # Convert the wt sequence to a list
        wt_list = list(self.wt_seq)
        # Loop over the number of batches
        for i, combo_batch in enumerate(np.array_split(sequences, n_batches)):
            # Create a list in which we will store SeqRecords
            temp_seqs = [None for _ in range(len(combo_batch))]
            # Build fasta for the batch
            for j, combo in enumerate(combo_batch):
                # Make a copy of the wild type list
                temp_seq = wt_list.copy()
                # Loop over the target python indices and set new amino acids
                for aa, ind in zip(combo, self.target_protein_inds):
                    # Replace the WT amino acids with the new ones
                    temp_seq[ind-1] = aa
                temp_seqs[j] = "".join(temp_seq)
            extracted_embeddings[i] = self._embed(temp_seqs)
        # Return the extracted embeddings, concatenating along the first dimension
        return np.concatenate(extracted_embeddings, axis=0)
    
    def _embed(
        self,
        sequences: Sequence[str],
        toks_per_batch: int = 4096,
        nogpu: bool = False,
    ) -> np.ndarray:
        model = deepcopy(self.pretrained_model)
        if torch.cuda.is_available() and not nogpu:
            model = self.pretrained_model.cuda()
        model.eval()
        ###
        labels = [None for _ in range(len(sequences))]
        dataset = FastaBatchedDataset(labels, sequences)
        batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
        data_loader = torch.utils.data.DataLoader(
            dataset, collate_fn=self.esm_alphabet.get_batch_converter(), batch_sampler=batches
        )
        ###
        results = []
        for _, _, toks in data_loader:
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            with torch.no_grad():
                out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
            representations = out["representations"][model.num_layers]
            results.append(representations[:, self.target_protein_inds].detach().cpu().numpy())
        return np.concatenate(results, axis=0)
