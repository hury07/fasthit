import os
import math
import numpy as np
import pandas as pd
import torch
from typing import Optional, Sequence, Tuple, Union
###
from argparse import Namespace
import warnings
import urllib
from pathlib import Path
try:
    import esm
except ImportError:
    pass
###

import fasthit

_filedir = os.path.dirname(__file__)
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

class ESM(fasthit.Encoder):
    def __init__(
        self,
        encoding: str,
        alphabet: str,
        wt_seq: str,
        target_python_idxs: Sequence[int],
        batch_size: int = 256,
        nogpu: bool = False,
        pretrained_model_dir: Optional[str] = None,
        database: Optional[str] = None,
        msa_depth: int = 64,
        msa_batch_size:  int = 8,
        n_threads: int = 8,
    ):
        try:
            esm
        except NameError as e:
            raise ImportError(
                "fair-esm not installed. "
                "Source code are available at "
                "https://github.com/facebookresearch/esm"
            ) from e

        assert encoding in ["esm-1b", "esm-1v", "esm-msa-1", "esm-msa-1b"]
        name = f"ESM_{encoding}"

        self._encoding = encodings.loc[encoding]
        self._wt_seq = wt_seq
        self._target_python_idxs = target_python_idxs
        self._embeddings = {}
        
        self._device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
        if pretrained_model_dir is None:
            pretrained_model_dir = _homedir + "/databases/pretrained_models/esm/"
        pretrained_model_file = pretrained_model_dir + self._encoding["model"]+".pt"
        if not os.path.isfile(pretrained_model_file):
            pretrained_model_file = self._encoding["model"]
        pretrained_model, esm_alphabet = Pretrained.load_model_and_alphabet(pretrained_model_file)
        ###
        self._pretrained_model = pretrained_model.eval().to(self._device)
        self._batch_converter = esm_alphabet.get_batch_converter()
        self._target_protein_idxs = [idx + 1 for idx in target_python_idxs]
        ### For MSA-Transformer
        if self._encoding.name in ["esm-msa-1", "esm-msa-1b"]:
            if database is None:
                database = _homedir + "/databases/hhsuite/uniclust30/UniRef30_2020_06"
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
            data = [(label, seq) for label, seq in zip(combo_batch, temp_seqs)]
            if self._encoding.name in ["esm-1b", "esm-1v"]:
                extracted_embeddings[i] = self._embed(data)
            elif self._encoding.name in ["esm-msa-1", "esm-msa-1b"]:
                extracted_embeddings[i] = self._embed_msa(data)
        unencoded = np.concatenate(extracted_embeddings, axis=0)
        embeddings = np.empty((len(sequences), *unencoded.shape[1:]), dtype=np.float32)
        for i, idx in enumerate(encoded_idx):
            embeddings[idx] = encoded[i]
        for i, idx in enumerate(unencoded_idx):
            embeddings[idx] = unencoded[i]
        return embeddings
    
    def _embed(
        self,
        data: Sequence[Union[Tuple, Sequence[Tuple]]],
    ) -> np.ndarray:
        labels, _, toks = self._batch_converter(data)
        toks = toks.to(self._device)
        with torch.no_grad():
            out = self._pretrained_model(
                toks, repr_layers=[self._pretrained_model.num_layers], return_contacts=False
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
        repr_dict = {labels[idx]: embedding[idx] for idx in range(embedding.shape[0])}
        self._embeddings.update(repr_dict)
        return embedding
    
    def _embed_msa(
        self,
        data: Sequence[Tuple],
    ) -> np.ndarray:
        ###
        import shutil
        import string
        from Bio import SeqIO
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        import subprocess
        from joblib import Parallel, delayed
        ###
        def _get_msa(name, seq, **kwargs):
            ###
            def _hamming_distance(s1, s2):
                #Return the Hamming distance between equal-length sequences
                if len(s1) != len(s2):
                    raise ValueError("Undefined for sequences of unequal length")
                return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))
            ###
            def _remove_insertions(sequence: str) -> str:
                deletekeys = dict.fromkeys(string.ascii_lowercase)
                deletekeys["."] = None
                deletekeys["*"] = None
                translation = str.maketrans(deletekeys)
                return sequence.translate(translation)
            def _filt_msa(file, msa_depth, temp_dir):
                labels = []
                seqs = []
                for record in SeqIO.parse(f"{temp_dir}/{file}.a3m", "fasta"):
                    labels.append(record.id)
                    seqs.append(_remove_insertions(str(record.seq)))
                ###
                seq_ref = seqs[0]
                label_ref = labels[0]
                seqs = seqs[1:]
                labels = labels[1:]
                dist = []
                for seq in seqs:
                    dist.append(_hamming_distance(seq, seq_ref))
                ###
                dist = np.array(dist)
                msa_depth = min(len(dist), msa_depth)
                sort_order = np.argsort(dist)[:-msa_depth-1:-1]
                return [(label_ref, seq_ref)] + [(labels[idx], seqs[idx]) for idx in sort_order]
            ###
            temp_dir = kwargs["temp_dir"]
            with open(f"{temp_dir}/{name}.fasta", "w") as f:
                SeqIO.write(SeqRecord(Seq(seq), id = name,  description=""), f, "fasta")
            ###
            _ = subprocess.run(
                [
                    "hhblits", "-i", f"{temp_dir}/{name}.fasta", "-oa3m", f"{temp_dir}/{name}.a3m",
                    "-d", kwargs["database"], "-v", "1", "-cpu", str(kwargs["n_threads"]),
                    "-diff", str(kwargs["msa_depth"])
                ]
            )
            ###
            return _filt_msa(name, kwargs["msa_depth"], temp_dir)
        ###
        if not os.path.exists(self._temp_dir):
            os.makedirs(self._temp_dir)
        kwargs = {
            "database": self._database,
            "msa_depth": self._msa_depth,
            "n_threads": self._n_threads,
            "temp_dir": self._temp_dir,
        }
        n_batches = math.ceil(len(data) / self._msa_batch_size)
        results = []
        with Parallel(n_jobs=self._msa_batch_size) as parallel:
            for batch in np.array_split(data, n_batches):
                msas = parallel(
                    delayed(_get_msa)(name, seq, **kwargs) for name, seq in batch
                )
                results.append(self._embed(msas))
        shutil.rmtree(self._temp_dir)
        return np.concatenate(results, axis=0)


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
            data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
        except RuntimeError:
            # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
            fn = Path(url).name
            data = torch.load(
                f"{torch.hub.get_dir()}/checkpoints/{fn}",
                map_location="cpu",
            )
        except urllib.error.HTTPError as e:
            raise Exception(f"Could not load {url}, check if you specified a correct model name?")
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
            pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
            prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
            prs2 = lambda s: "".join(
                s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
            )
            model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
            model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
            model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
            model_args["emb_layer_norm_before"] = cls.has_emb_layer_norm_before(model_state)
            model_type = esm.ProteinBertModel

        elif model_data["args"].arch == "protein_bert_base":

            # upgrade state dict
            pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
            prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
            model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
            model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
            model_type = esm.ProteinBertModel
        elif model_data["args"].arch == "msa_transformer":

            # upgrade state dict
            pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
            prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
            prs2 = lambda s: "".join(
                s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
            )
            prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
            model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
            model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
            if model_args.get("embed_positions_msa", False):
                emb_dim = model_state["msa_position_embedding"].size(-1)
                model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1

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
            expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
            error_msgs = []
            missing = (expected_keys - found_keys) - expected_missing
            if missing:
                error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
            unexpected = found_keys - expected_keys
            if unexpected:
                error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

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
