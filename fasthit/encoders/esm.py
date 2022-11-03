import math
import os
import urllib
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from fasthit import Encoder
from fasthit.error import EsmImportError
from fasthit.utils.dataset import SequenceData, collate_batch
from torch.utils.data import DataLoader

try:
    import esm
except ImportError:
    pass
###

_homedir = os.path.expanduser("~")

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


class ESM(Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
        pretrained_model_dir: Optional[str] = None,
    ):
        try:
            esm
        except Exception:
            raise EsmImportError

        assert encoding in ["esm-1b", "esm-1v"]
        name = f"ESM_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._target_python_idxs = target_python_idxs
        self._embeddings: Dict[str, np.ndarray] = {}
        self._wt_list = list(wt_seq)

        self._device = torch.device(
            'cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')

        self._load_model(target_python_idxs, pretrained_model_dir)

        super().__init__(
            name,
            alphabet,
            self._encoding["n_features"],
            batch_size=batch_size
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

    def encode_func(self, sequences: Sequence[str],
                    embed: Callable[[Sequence[str], Sequence[str]], np.ndarray]) -> np.ndarray:
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
        n_batches = math.ceil(len(unencoded_seqs) / self.batch_size)
        ###
        extracted_embeddings: List[np.ndarray] = [
            None for _ in range(n_batches)]

        dataset = SequenceData(
            unencoded_seqs, self._target_python_idxs, self._wt_list)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_batch)

        for i, data in enumerate(dataloader):
            extracted_embeddings[i] = embed(data)
        unencoded = np.concatenate(extracted_embeddings, axis=0)
        embeddings = np.empty(
            (len(sequences), *unencoded.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(encoded_idx):
            embeddings[idx] = encoded[i]
        for i, idx in enumerate(unencoded_idx):
            embeddings[idx] = unencoded[i]
        return embeddings

    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        return self.encode_func(sequences, self._embed)

    @torch.no_grad()
    def _embed(
        self,
        data: Sequence[Union[Tuple, Sequence[Tuple]]],
    ) -> np.ndarray:
        labels, _, toks = self._batch_converter(data)
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
            embedding = embedding[:, 0]
            labels = labels[0]
        embedding = embedding[:, self._target_protein_idxs]
        repr_dict = {labels[idx]: embedding[idx]
                     for idx in range(embedding.shape[0])}
        self._embeddings.update(repr_dict)
        return embedding


class Pretrained(object):
    # Copyright (c) Facebook, Inc. and its affiliates.
    #
    # This source code is licensed under the MIT license found in the
    # LICENSE file in the root directory of this source tree.
    def __init__(self) -> None:
        pass

    @classmethod
    def load_model_and_alphabet(cls, model_name):
        if model_name.endswith(".pt"):  # treat as filepath
            return cls.load_model_and_alphabet_local(model_name)
        else:
            return cls.load_model_and_alphabet_hub(model_name)

    @classmethod
    def load_model_and_alphabet_local(cls, model_location):
        """ Load from local path. The regression weights need to be co-located """
        model_location = Path(model_location)
        model_data = torch.load(model_location, map_location="cpu")
        regression_data = None
        return cls.load_model_and_alphabet_core(model_data, regression_data)

    @classmethod
    def load_model_and_alphabet_hub(cls, model_name):
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = cls.load_hub_workaround(url)
        regression_data = None
        return cls.load_model_and_alphabet_core(model_data, regression_data)

    @classmethod
    def load_hub_workaround(cls, url):
        try:
            data = torch.hub.load_state_dict_from_url(
                url, progress=False, map_location="cpu")
        except urllib.error.HTTPError as e:
            raise Exception(
                f"Could not load {url}, check if you specified a correct model name?")
        return data

    @classmethod
    def has_emb_layer_norm_before(cls, model_state):
        """ Determine whether layer norm needs to be applied before the encoder """
        return any(k.startswith("emb_layer_norm_before") for k, param in model_state.items())

    @classmethod
    def load_model_and_alphabet_core(cls, model_data, regression_data=None):
        if regression_data is not None:
            model_data["model"].update(regression_data["model"])

        alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

        if model_data["args"].arch == "roberta_large":
            # upgrade state dict
            def pra(s): return "".join(
                s.split("encoder_")[1:] if "encoder" in s else s)

            def prs1(s): return "".join(
                s.split("encoder.")[1:] if "encoder" in s else s)

            def prs2(s): return "".join(
                s.split("sentence_encoder.")[
                    1:] if "sentence_encoder" in s else s
            )
            model_args = {pra(arg[0]): arg[1]
                          for arg in vars(model_data["args"]).items()}
            model_state = {prs1(prs2(arg[0])): arg[1]
                           for arg in model_data["model"].items()}
            # For token drop
            model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()
            model_args["emb_layer_norm_before"] = cls.has_emb_layer_norm_before(
                model_state)
            model_type = esm.ProteinBertModel

        elif model_data["args"].arch == "protein_bert_base":

            # upgrade state dict
            def pra(s): return "".join(
                s.split("decoder_")[1:] if "decoder" in s else s)

            def prs(s): return "".join(
                s.split("decoder.")[1:] if "decoder" in s else s)
            model_args = {pra(arg[0]): arg[1]
                          for arg in vars(model_data["args"]).items()}
            model_state = {prs(arg[0]): arg[1]
                           for arg in model_data["model"].items()}
            model_type = esm.ProteinBertModel
        elif model_data["args"].arch == "msa_transformer":

            # upgrade state dict
            def pra(s): return "".join(
                s.split("encoder_")[1:] if "encoder" in s else s)

            def prs1(s): return "".join(
                s.split("encoder.")[1:] if "encoder" in s else s)

            def prs2(s): return "".join(
                s.split("sentence_encoder.")[
                    1:] if "sentence_encoder" in s else s
            )

            def prs3(s): return s.replace(
                "row", "column") if "row" in s else s.replace("column", "row")
            model_args = {pra(arg[0]): arg[1]
                          for arg in vars(model_data["args"]).items()}
            model_state = {prs1(prs2(prs3(arg[0]))): arg[1]
                           for arg in model_data["model"].items()}
            if model_args.get("embed_positions_msa", False):
                emb_dim = model_state["msa_position_embedding"].size(-1)
                # initial release, bug: emb_dim==1
                model_args["embed_positions_msa_dim"] = emb_dim

            model_type = esm.MSATransformer

        else:
            raise ValueError("Unknown architecture selected")

        model = model_type(
            Namespace(**model_args),
            alphabet,
        )

        expected_keys = set(model.state_dict().keys())
        found_keys = set(model_state.keys())

        if regression_data is None:
            expected_missing = {
                "contact_head.regression.weight", "contact_head.regression.bias"}
            error_msgs = []
            missing = (expected_keys - found_keys) - expected_missing
            if missing:
                error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
            unexpected = found_keys - expected_keys
            if unexpected:
                error_msgs.append(
                    f"Unexpected key(s) in state_dict: {unexpected}.")

            if error_msgs:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
            if expected_missing - found_keys:
                warnings.warn(
                    "Regression weights not found, predicting contacts will not produce correct results."
                )

        model.load_state_dict(model_state, strict=regression_data is not None)

        return model, alphabet
