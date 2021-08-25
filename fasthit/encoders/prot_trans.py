import re
import math
import numpy as np
import pandas as pd
import torch
from typing import Sequence

import fasthit

encodings = pd.DataFrame(
    {
        "encoder": ["prot_bert_bfd", "prot_t5_xl_uniref50"],
        "n_features": [1024, 1024],
    }
)
encodings.set_index("encoder", inplace=True)

class ProtTrans(fasthit.Encoder):
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
        self._encoding = encodings.loc[encoding]
        self._wt_seq = wt_seq
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}

        self._device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
        if encoding in ["prot_bert_bfd"]:
            try:
                from transformers import BertModel, BertTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers which contained ProtTrans not installed. "
                    "Source code are available at "
                    "https://github.com/huggingface/transformers"
                ) from e
            self._tokenizer = BertTokenizer.from_pretrained(f"Rostlab/{encoding}", do_lower_case=False)
            self._pretrained_model = BertModel.from_pretrained(f"Rostlab/{encoding}")
            self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        elif encoding in ["prot_t5_xl_uniref50"]:
            try:
                from transformers import T5EncoderModel, T5Tokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers which contained ProtTrans not installed. "
                    "Source code are available at "
                    "https://github.com/huggingface/transformers"
                ) from e
            self._tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{encoding}", do_lower_case=False)
            self._pretrained_model = T5EncoderModel.from_pretrained(f"Rostlab/{encoding}")
            self._target_protein_idxs = target_python_idxs
        else:
            pass
        self._pretrained_model = self._pretrained_model.to(self._device)
        self._pretrained_model.eval()
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
        ###
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
                temp_seqs[j] = " ".join(temp_seq)
            temp_seqs = [re.sub(r"[UZOB]", "X", seq) for seq in temp_seqs]
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
        ids = self._tokenizer.batch_encode_plus(
            sequences, add_special_tokens=True, pad_to_max_length=True
        )
        input_ids = torch.tensor(ids['input_ids']).to(self._device)
        attention_mask = torch.tensor(ids['attention_mask']).to(self._device)
        with torch.no_grad():
            embeddings = self._pretrained_model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = embeddings[:, self._target_protein_idxs].cpu().numpy()
        del input_ids, attention_mask, embeddings
        torch.cuda.empty_cache()
        ###
        repr_dict = {labels[idx]: embedding[idx] for idx in range(embedding.shape[0])}
        self._embeddings.update(repr_dict)
        return embedding
