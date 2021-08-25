###
import math
from typing import Sequence, Union

import numpy as np
import torch
from esm import pretrained
from fasthit.encoders.esm import encodings

import fasthit


class ExtendHalf(fasthit.Encoder):
    def __init__(self,
                 encoding: str,
                 wt_seq: Sequence[str],
                 target_protein_idxs: Sequence[int],
                 pretrained_model_exprs: Sequence[int], 
                 pretrained_model_dir: str,
                 nogpu: bool = False):

        assert encoding in ["esm-1b", "esm-1v"]
        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')

        pretrained_model_name = encodings.loc[encoding]["model"]
        self._pretrained_model, self.esm_alphabet = pretrained.load_model_and_alphabet(
            pretrained_model_dir + pretrained_model_name + ".pt"
        )
        self._pretrained_model = self._pretrained_model.to(self._device)

        self.pretrained_model_exprs = pretrained_model_exprs

        self._target_protein_idxs = target_protein_idxs
        self._wt_seq = list(wt_seq)
        self._embeddings = {}
        

        super().__init__("onehot", self.esm_alphabet.all_toks, len(self.esm_alphabet))

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
                for aa, ind in zip(combo, self._target_protein_idxs):
                    temp_seq[ind] = aa
                temp_seqs[j] = "".join(temp_seq)
            extracted_embeddings[i] = self._embed(temp_seqs, combo_batch)
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
            dataset, collate_fn=self.esm_alphabet.get_batch_converter(), batch_sampler=batches
        )
        ###
        results = []
        for labels, _, toks in data_loader:
            toks = toks.to(self._device)
            with torch.no_grad():
                out = self._pretrained_model(
                    toks, repr_layers=self.pretrained_model_exprs, return_contacts=False
                )
            embeddings = out["representations"]
            embeddings = torch.stack([v for _, v in embeddings.items()], dim=-1)
            embedding = embeddings.detach().cpu().numpy()
            del toks, embeddings
            torch.cuda.empty_cache()
            ###
            embedding = embedding[:, self._target_protein_idxs ,: ,  :]
            
            results.append(embedding)
            repr_dict = {labels[idx]: embedding[idx]
                         for idx in range(embedding.shape[0])}
            self._embeddings.update(repr_dict)
        return np.concatenate(results, axis=0)
