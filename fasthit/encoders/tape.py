###
from typing import Sequence

import math
import numpy as np
import pandas as pd
import torch

import fasthit


encodings = pd.DataFrame(
    {
        "encoder": ["transformer", "unirep", "trrosetta"],
        "model": ["bert-base", "babbler-1990", ["xaa", "xab", "xac", "xad", "xae"]],
        "n_features": [768, 1900, 526],
    }
)
encodings.set_index("encoder", inplace=True)


class TAPE(fasthit.Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
    ):
        assert encoding in ["transformer", "unirep", "trrosetta"]

        name = f"TAPE_{encoding}"
        self._encoding = encodings.loc[encoding]
        self._wt_seq = wt_seq
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}

        self._device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
        import tape
        self._tokenizer = tape.TAPETokenizer(vocab='iupac')
        if self._encoding["model"] in ["bert-base"]:
            self._pretraind_model = tape.ProteinBertModel.from_pretrained(self._encoding["model"])
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        elif self._encoding["model"] in ["babbler-1900"]:
            self._pretraind_model = tape.UniRepModel.from_pretrained(self._encoding["model"])
            self._tokenizer = tape.TAPETokenizer(vocab='unipre')
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs] # TODO check
        elif self._encoding["model"] in ["xaa", "xab", "xac", "xad", "xae"]:
            self._pretraind_model = tape.TRRosetta.from_pretrained(self._encoding["model"])
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs] # TODO check
        self._pretraind_model = self._pretraind_model.to(self._device)
        self._pretraind_model.eval()

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size
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
            extracted_embeddings[i] = self._embed(temp_seqs, combo_batch)
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
    ) -> np.ndarray:
        token_ids = torch.tensor(
            [self._tokenizer.encode(seq) for seq in sequences], device=self._device
        )
        with torch.no_grad():
            embeddings, _ = self._pretraind_model(token_ids)
        embeddings = embeddings[:, self._target_protein_idxs].detach().cpu().numpy()
        repr_dict = {labels[idx]: embeddings[idx] for idx in range(embeddings.shape[0])}
        self._embeddings.update(repr_dict)
        torch.cuda.empty_cache()
        return embeddings