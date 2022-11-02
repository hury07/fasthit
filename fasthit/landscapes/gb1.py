"""Define GB1 landscape and problem registry."""
from typing import Dict, Sequence
import os
import numpy as np
import pandas as pd

import fasthit

_filedir = os.path.dirname(__file__)

class GB1(fasthit.Landscape):
    """
    A landscape of stability of protein G domain B1 and the bidxsing affinity to IgG-Fc.

    We use experimental and imputed data from Wu et al. (2016).
    """
    _wt = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    # Variants (V39, D40, G41 and V54)
    # Fitness_wt: VDGV - 1.0
    # Fitness_max (measured): FWAA - 8.76196565571
    # Fitness_max:            AHCA - 9.9130883273888

    def __init__(self, data_used, search_space):
        """
        Create a GB1 landscape from data .csv files.
        """
        super().__init__(name="GB1_combo")

        if data_used == "trainset":
            data = pd.read_csv(
                os.path.join(
                    _filedir, "data/gb1/trainset.csv"
                )
            )
            norm_score = data["Fitness"]
        elif data_used == "testset":
            data = pd.read_csv(
                os.path.join(
                    _filedir, "data/gb1/testset.csv"
                )
            )
            norm_score = data["Fitness"]
        else:
            # Load GB1 measurements from file
            measured_data = pd.read_csv(
                os.path.join(
                    _filedir, "data/gb1/elife-16965-supp1-v4.csv"
                )
            )
            measured_data = measured_data[["Variants", "Fitness"]]
            if data_used == "measured_only":
                data = measured_data
            else:
                imputed_data = pd.read_csv(
                    os.path.join(
                        _filedir, "data/gb1/elife-16965-supp2-v4.csv"
                    )
                )
                imputed_data.columns = ["Variants", "Fitness"]
                if data_used == "imputed_only":
                    data = imputed_data
                elif data_used == "with_imputed":
                    data = pd.concat([measured_data, imputed_data])
            score = data["Fitness"] # WT score is set to 1.
            norm_score = (score - score.min()) / (score.max() - score.min())

        self._sequences = dict(zip(data["Variants"], norm_score))

        combo = search_space.split(",")
        temp_seq = [idxs[0] for idxs in combo]
        self._combo_protein_idxs = [int(idxs[1:]) for idxs in combo]
        self._combo_python_idxs = [idxs - 1 for idxs in self._combo_protein_idxs]
        
        assert all([
            self._wt[self._combo_python_idxs[i]] == temp_seq[i] for i in range(len(temp_seq))
        ])

    def _fitness_function(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array(
            [self._sequences.get(seq, np.nan) for seq in sequences],
            dtype=np.float32
        )
    
    @property
    def wt(self):
        return self._wt
    
    @property
    def combo_protein_idxs(self):
        return self._combo_protein_idxs
    
    @property
    def combo_python_idxs(self):
        return self._combo_python_idxs


def registry() -> Dict[str, Dict]:
    """
    Return a dictionary of problems of the form:

    ```python
    {
        "problem name": {
            "params": ...,
            "starts": ...,
        },
        ...
    }
    ```

    where `fasthit.landscapes.GB1(**problem["params"])` instantiates the
    GB1 landscape for the given set of parameters.

    Returns:
        Problems in the registry.
    """
    wild_type = [
        "VDGV",
    ]
    measured_only = [
        "FWAL",
        "FWAM",
        "FWAI",
        "FEAA",
        "RNAA",
        "QWFA",
        "FWLV",
        "ECAA",
        "FDGA",
        "FMCV",
        "LWGG",
        "MFGA",
        "STCA",
        "MCMA",
        "TFGS",
        "QRCC",
        "MIGS",
        "ITLG",
        "HSFG",
    ]
    with_imputed = [
        "AHCF",
        "PHCA",
        "AHLA",
        "AHGA",
        "NHCA",
        "HHCM",
        "PNCA",
        "AAFA",
        "KHCG",
        "ADAA",
        "FACV",
        "VAYA",
        "NHGL",
        "TKAA",
        "LACC",
        "DCQF",
        "MVAL",
        "MSGT",
        "YCGT",
        "LRAG",
    ]
    trainset = [
        "FWAL",
        "FWAM",
        "FWAI",
        "FEAA",
        "RNAA",
        "RIAA",
        "FWLV",
        "ECAA",
        "FDGA",
        "FMGL",
        "LWGQ",
        "MWGS",
        "STCA",
        "MCMA",
        "TFGY",
        "QSCC",
        "MKGM",
        "IVYG",
        "ICCG",
    ]
    testset = []

    problems = {
        "measured_only": {
            "params": {
                "data_used": "measured_only",
                "search_space": "V39,D40,G41,V54",
            },
            "wt_only": wild_type,
            "starts": measured_only,
        },
        "with_imputed": {
            "params": {
                "data_used": "with_imputed",
                "search_space": "V39,D40,G41,V54",
            },
            "wt_only": wild_type,
            "starts": with_imputed,
        },
        "trainset": {
            "params": {
                "data_used": "trainset",
                "search_space": "V39,D40,G41,V54",
            },
            "starts": trainset,
        },
        "testset": {
            "params": {
                "data_used": "testset",
                "search_space": "V39,D40,G41,V54",
            },
            "starts": testset,
        }
    }

    return problems
