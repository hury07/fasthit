###
from copy import deepcopy
from typing import Sequence

import math
import numpy as np
import pandas as pd
import torch

import tape
import fasthit


class TAPE(fasthit.Encoder):
    encodings = pd.DataFrame(
        {
            "encoder": ["transformer", "unirep", "trrosetta"],
            "model": ["bert-base", "babbler-1990", ["xaa", "xab", "xac", "xad", "xae"]],
            "n_features": [768, 1900, 526],
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
    ):
        assert encoding in ["transformer", "unirep", "trrosetta"]

        name = f"tape_{encoding}"
        self.encoding = self.encodings.loc[encoding]
        self.wt_seq = wt_seq
        self.target_protein_inds = target_protein_inds
        self.n_positions_combined = len(target_protein_inds)

        self.tokenizer = tape.TAPETokenizer(vocab='iupac')
        if self.encoding["model"] in ["bert-base"]:
            self.pretraind_model = tape.ProteinBertModel.from_pretrained(self.encoding["model"]) 
        elif self.encoding["model"] in ["babbler-1900"]:
            self.pretraind_model = tape.UniRepModel.from_pretrained(self.encoding["model"])
            self.tokenizer = tape.TAPETokenizer(vocab='unipre')
        elif self.encoding["model"] in ["xaa", "xab", "xac", "xad", "xae"]:
            self.pretraind_model = tape.TRRosetta.from_pretrained(self.encoding["model"])

        super().__init__(
            name,
            alphabet,
            self.encoding["n_features"],
            batch_size=batch_size
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
        nogpu: bool = False,
    ) -> np.ndarray:
        token_ids = torch.tensor([self.tokenizer.encode(seq) for seq in sequences])
        model = deepcopy(self.pretraind_model)
        if torch.cuda.is_available() and not nogpu:
            model = self.pretraind_model.cuda()
            token_ids = token_ids.to(device="cuda", non_blocking=True)
        model.eval()
        with torch.no_grad():
            seq_embed, _ = model(token_ids)
        return seq_embed[:, self.target_protein_inds].detach().cpu().numpy()