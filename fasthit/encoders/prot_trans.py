import re
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from fasthit import Encoder
from fasthit.error import TransformerImportError
from fasthit.utils.dataset import SequenceData, collate_batch
from torch import nn
from torch.utils.data import DataLoader

encodings = pd.DataFrame(
    {
        "encoder": ["prot_bert_bfd", "prot_t5_xl_uniref50"],
        "n_features": [1024, 1024],
    }
)
encodings.set_index("encoder", inplace=True)


class ProtTrans(Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
    ):
        assert encoding in ["prot_bert_bfd", "prot_t5_xl_uniref50"]
        name = f"ProtTrans_{encoding}"

        name = f"ProtTrans_{encoding}"
        self._encoding = encodings.loc[encoding]
        self._wt_list = list(wt_seq)
        self._embeddings = {}
        self._target_protein_idxs: List[int] = []
        self._target_python_idxs = target_python_idxs
        self._pretrained_model: nn.Module

        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')

        self._load_model(encoding, target_python_idxs)

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size
        )

    def _load_model(self, encoding: str, target_python_idxs: Sequence[int],):
        if encoding in {"prot_bert_bfd"}:
            try:
                from transformers import BertModel, BertTokenizer
            except ImportError:
                raise TransformerImportError
            self._tokenizer = BertTokenizer.from_pretrained(
                f"Rostlab/{encoding}", do_lower_case=False)
            self._pretrained_model = BertModel.from_pretrained(
                f"Rostlab/{encoding}")
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        elif encoding in {"prot_t5_xl_uniref50"}:
            try:
                from transformers import T5EncoderModel, T5Tokenizer
            except ImportError:
                raise TransformerImportError
            self._tokenizer = T5Tokenizer.from_pretrained(
                f"Rostlab/{encoding}", do_lower_case=False)
            self._pretrained_model = T5EncoderModel.from_pretrained(
                f"Rostlab/{encoding}")
            self._target_protein_idxs = target_python_idxs
        else:
            pass
        self._pretrained_model = self._pretrained_model.to(self._device)
        self._pretrained_model.eval()

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        ### [bsz, seq_length, n_features]
        encoded_idx = [
            idx for idx in range(len(sequences)) if sequences[idx] in self._embeddings.keys()
        ]
        encoded = [self._embeddings[sequences[idx]] for idx in encoded_idx]
        if len(encoded_idx) == len(sequences):
            return np.array(encoded, dtype=np.float32)

        unencoded_idx = [idx for idx in range(
            len(sequences)) if idx not in encoded_idx]
        unencoded_seqs = [sequences[idx] for idx in unencoded_idx]

        dataset = SequenceData(
            unencoded_seqs, self._target_python_idxs, self._wt_list)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, collate_fn=collate_batch)

        extracted_embeddings: List[np.ndarray] = [
            None for _ in range(len(dataloader))]
        for i, (combo_batch, temp_seqs) in enumerate(dataloader):
            temp_seqs = [re.sub(r"[UZOB]", "X", seq) for seq in temp_seqs]
            extracted_embeddings[i] = self._embed(temp_seqs, combo_batch)
        unencoded = np.concatenate(extracted_embeddings, axis=0)

        embeddings = np.empty(
            (len(sequences), *unencoded.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(encoded_idx):
            embeddings[idx] = encoded[i]
        for i, idx in enumerate(unencoded_idx):
            embeddings[idx] = unencoded[i]
        return embeddings

    @torch.no_grad()
    def _embed(
        self,
        sequences: Sequence[str],
        labels: Sequence[str],
    ) -> np.ndarray:
        ids = self._tokenizer.batch_encode_plus(
            sequences, add_special_tokens=True, pad_to_max_length=True
        )
        input_ids = torch.tensor(ids['input_ids']).to(self._device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self._device)

        embeddings = self._pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embeddings[:, self._target_protein_idxs].cpu().numpy()
        del input_ids, attention_mask, embeddings
        torch.cuda.empty_cache()

        repr_dict = {labels[idx]: embedding[idx]
                     for idx in range(embedding.shape[0])}
        self._embeddings.update(repr_dict)
        return embedding
