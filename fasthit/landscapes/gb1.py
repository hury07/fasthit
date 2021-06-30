"""Define GB1 landscape and problem registry."""
import os
from typing import Dict, Sequence

import numpy as np
import pandas as pd

import fasthit

class GB1(fasthit.Landscape):
    """
    A landscape of stability of protein G domain B1 and the bidxsing affinity to IgG-Fc.

    We use experimental and imputed data from Wu et al. (2016).
    """
    _wt = (
        "MTYKLILNGK"
        "TLKGETTTEA"
        "VDAATAEKVF"
        "KQYANDNGVD"
        "GEWTYDDATK"
        "TFTVTE"
    )
    # Variants (V39, D40, G41 and V54)
    # Fitness_wt: VDGV - 1.0
    # Fitness_max (measured): FWAA - 8.76196565571
    # Fitness_max:            AHCA - 9.9130883273888

    def __init__(self, data_used, search_space):
        """
        Create a GB1 landscape from data .xlsx files.
        """
        super().__init__(name="GB1_combo")

        # Load GB1 measurements from file
        measured_data = pd.read_excel(
            os.path.join(
                os.path.dirname(__file__),
                "data/gb1/elife-16965-supp1-v4.xlsx"
            )
        )
        measured_data = measured_data[["Variants", "Fitness"]]
        if data_used == "with_imputed":
            imputed_data = pd.read_excel(
                os.path.join(
                    os.path.dirname(__file__),
                    "data/gb1/elife-16965-supp2-v4.xlsx"
                )
            )
            imputed_data.columns = ["Variants", "Fitness"]
            data = pd.concat([measured_data, imputed_data])
        elif data_used == "only_measured":
            data = measured_data

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
            [self._sequences[seq] for seq in sequences],
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
    only_measured = [
        "FWRA",
        "FWWA",
        "FWSA",
        "FEAA",
        "KWAY",
        "FTAY",
        "FCWA",
        "MWSA",
        "FHSA",
        "KLNA",
        "SNVA",
        "IYAN",
        "WEGA",
        "LLLA",
        "EVWL",
        "LFDI",
        "VYFL",
        "YFCI",
        "VYVG",
    ]

    with_imputed = [
        "AHNA",
        "CHCA",
        "NHCA",
        "ATRA",
        "KAHC",
        "AHHK",
        "CRCA",
        "AHGC",
        "WHRH",
        "GSCQ",
        "AGFR",
        "YYCS",
        "AMLG",
        "SIDW",
        "CMPW",
        "VRFM",
        "KVGF",
        "MRGM",
    ]

    problems = {
        "only_measured": {
            "params": {
                "data_used": "only_measured",
                "search_space": "V39,D40,G41,V54",
            },
            "starts": only_measured,
        },
        "with_imputed": {
            "params": {
                "data_used": "with_imputed",
                "search_space": "V39,D40,G41,V54",
            },
            "starts": with_imputed,
        },
    }

    return problems
