from typing import Callable, List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from fasthit import Encoder
from fasthit.error import TapeImportError
from fasthit.utils.dataset import SequenceData, collate_batch
from torch.utils.data import DataLoader

encodings = pd.DataFrame(
    {
        "encoder": ["transformer", "unirep"],
        "model": ["bert-base", "babbler-1900"],
        "n_features": [768, 1900],
    }
)
encodings.set_index("encoder", inplace=True)


class TAPE(Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False
    ):
        assert encoding in ["transformer", "unirep"]
        name = f"TAPE_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_list = list(wt_seq)
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}

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

        self._tokenizer = tape.TAPETokenizer(vocab='iupac')
        if self._encoding["model"] in ["bert-base"]:
            pretrained_model = tape.ProteinBertModel.from_pretrained(
                self._encoding["model"])
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        elif self._encoding["model"] in ["babbler-1900"]:
            pretrained_model = tape.UniRepModel.from_pretrained(
                self._encoding["model"])
            self._tokenizer = tape.TAPETokenizer(vocab='unirep')
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]

        self._pretrained_model = pretrained_model.to(self._device)
        self._pretrained_model.eval()

    def encode_func(self, sequences: Sequence[str],
                    embed: Callable[[Sequence[str], Sequence[str]], np.ndarray], collate_fn: Union[Callable, None]) -> np.ndarray:
        ### [bsz, seq_length, n_features]
        # cache for already encoded sequences
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
        ###
        extracted_embeddings: List[np.ndarray] = [
            None for _ in range(len(dataloader))]

        # Loop over the number of batches
        dataset = SequenceData(
            unencoded_seqs, self._target_python_idxs, self._wt_list)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn)

        for i, (combo_batch, temp_seqs) in enumerate(dataloader):
            extracted_embeddings[i] = embed(temp_seqs, combo_batch)

        unencoded = np.concatenate(extracted_embeddings, axis=0)
        embeddings = np.empty(
            (len(sequences), *unencoded.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(encoded_idx):
            embeddings[idx] = encoded[i]
        for i, idx in enumerate(unencoded_idx):
            embeddings[idx] = unencoded[i]
        return embeddings

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        self.encode_func(sequences, self._embed, collate_batch)

    @torch.no_grad()
    def _embed(
        self,
        sequences: Sequence[str],
        labels: Sequence[str],
    ) -> np.ndarray:
        token_ids = torch.tensor(
            [self._tokenizer.encode(seq) for seq in sequences], device=self._device
        )
        embeddings, _ = self._pretrained_model(token_ids)
        embedding = embeddings[:,
                               self._target_protein_idxs].detach().cpu().numpy()
        del token_ids, embeddings
        torch.cuda.empty_cache()
        ###
        repr_dict = {labels[idx]: embedding[idx]
                     for idx in range(embedding.shape[0])}
        self._embeddings.update(repr_dict)
        return embedding
