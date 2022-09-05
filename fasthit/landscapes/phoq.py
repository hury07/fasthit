"""Define PhoQ landscape and problem registry."""
from typing import Dict, Sequence
import os
import numpy as np
import pandas as pd

import fasthit

_filedir = os.path.dirname(__file__)

class PhoQ(fasthit.Landscape):
    """
    A landscape of stability of protein G domain B1 and the bidxsing affinity to IgG-Fc.

    We use experimental and imputed data from Wu et al. (2016).
    """
    _wt = (
        "MKKLLRLFFPLSLRVRFLLATAAVVLVLSLAYGMVALIGYSVSFDKTTFRLLRGESNLFY"
        "TLAKWENNKLHVELPENIDKQSPTMTLIYDENGQLLWAQRDVPWLMKMIQPDWLKSNGFH"
        "EIEADVNDTSLLLSGDHSIQQQLQEVREDDDDAEMTHSVAVNVYPATSRMPKLTIVVVDT"
        "IPVELKSSYMVWSWFIYVLSANLLLVIPLLWVAAWWSLRPIEALAKEVRELEEHNRELLN"
        "PATTRELTSLVRNLNRLLKSERERYDKYRTTLTDLTHSLKTPLAVLQSTLRSLRSEKMSV"
        "SDAEPVMLEQISRISQQIGYYLHRASMRGGTLLSRELHPVAPLLDNLTSALNKVYQRKGV"
        "NISLDISPEISFVGEQNDFVEVMGNVLDNACKYCLEFVEISARQTDEHLYIVVEDDGPGI"
        "PLSKREVIFDRGQRVDTLRPGQGVGLAVAREITEQYEGKIVAGESMLGGARMEVIFGRQH"
        "SAPKDE"
    )
    # Variants (A284, V285, S288 and T289)
    # Fitness_wt: AVST - 3.28744733333
    # Fitness_max (measured): TEMH - 133.59427

    def __init__(self, search_space):
        """
        Create a GB1 landscape from data .csv files.
        """
        super().__init__(name="PhoQ_combo")

        # Load GB1 measurements from file
        data = pd.read_csv(
            os.path.join(
                _filedir, "data/phoq/PhoQ.csv"
            )
        )

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
        "AVST",
    ]
    measured_only = [
        "TEMN",
        "TEMD",
        "TEQH",
        "TMMH",
        "TYMN",
        "TNMR",
        "RCMH",
        "TQMS",
        "TSYH",
        "HETI",
        "SCIH",
        "SEGL",
        "TQGY",
        "LAMN",
        "LGCL",
        "SMGS",
        "GGLQ",
        "AMCN",
        "MQLQ",
    ]

    problems = {
        "measured_only": {
            "params": {
                "search_space": "A284,V285,S288,T289",
            },
            "wt_only": wild_type,
            "starts": measured_only,
        },
    }

    return problems
