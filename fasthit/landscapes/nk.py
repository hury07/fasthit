"""Define NK landscape and problem registry."""
from typing import Sequence, Dict
import os
import itertools
import numpy as np

import fasthit
import fasthit.utils.sequence_utils as s_utils

_filedir = os.path.dirname(__file__)

class NK(fasthit.Landscape):
    """
    A mathmatical fitness landscape introduced by Stuart & Edward (1989)
    to simulating different levels of "ruggedness".
    """

    def __init__(
        self,
        N,
        K,
        epi='uni',
        pos_weight='one',
        cut_off=None,
        alphabet=s_utils.AAS,
        seed=42
    ):
        """
        Generate NK landscape with senquence length of N,
        and ruggedness level of K.
        """
        assert K < N
        assert epi in ['uni', 'exp', 'zipf']
        assert pos_weight in ['one', 'uni', 'exp', 'zipf']
        #
        super().__init__(name=f"N{N}K{K}")
        #
        self.alphabet = alphabet
        #
        dir_nk = _filedir + f"/data/nk"
        if not os.path.exists(dir_nk):
            os.makedirs(dir_nk)
        #
        if cut_off is None:
            path = dir_nk + f"/N{N}K{K}_epi_{epi}_weight_{pos_weight}_seed{seed}.npy"
        else:
            path = dir_nk + f"/N{N}K{K}_epi_{epi}_weight_{pos_weight}_cutoff{cut_off}_seed{seed}.npy"
        #
        if os.path.exists(path):
            self._sequences = np.load(path, allow_pickle=True).item()
        else:
            self._epi = epi
            self._rng = np.random.default_rng(seed)
            #
            if pos_weight == 'one':
                weight = np.ones(N)
            elif pos_weight == 'uni':
                weight = self._rng.uniform(0, 1, size=N)
            elif pos_weight == 'exp':
                weight = self._rng.exponential(scale=1., size=N)
            elif pos_weight == 'zipf':
                weight = self._rng.zipf(a=2, size=N).astype(float)
            weight /= weight.sum()
            #
            f_mem = {}
            epi_net = self.genEpiNet(N, K)
            sequenceSpace = np.array(list(itertools.product(alphabet, repeat=N)))
            score = np.array([
                self.fitness(i, epi=epi_net, mem=f_mem, w=weight) for i in sequenceSpace
            ])
            norm_score = (score - score.min()) / (score.max() - score.min())
            if cut_off is not None:
                score = np.where(norm_score > cut_off, norm_score, cut_off)
                norm_score = (score - score.min()) / (score.max() - score.min())
            #
            seqs = ["".join(seq) for seq in sequenceSpace]
            self._sequences = dict(zip(seqs, norm_score))
            #
            np.save(path, self._sequences)


    def _fitness_function(self, sequences: Sequence[str]) -> np.ndarray:
        return np.array(
            [self._sequences[seq] for seq in sequences],
            dtype=np.float32
        )
    
    def genEpiNet(self, N, K):
        """
        Generates a random epistatic network for a sequence of length
        N with, on average, K connections
        """
        return {
            i: sorted(self._rng.choice(
                [n for n in range(N) if n != i],
                K,
                replace=False
            ).tolist() + [i])
            for i in range(N)
        }
    
    def fitness(self, sequence, epi, mem, w):
        """
        Obtains a fitness value for the entire sequence by summing
        over individual amino acids
        """
        return np.mean([
            self.fitness_i(sequence, i, epi, mem) * w[i]
            for i in range(len(sequence))
        ])

    def fitness_i(self, sequence, i, epi, mem):
        """
        Assigns a (random) fitness value to the ith amino acid that
        interacts with K other positions in a sequence.
        """
        #we use the epistasis network to work out what the relation is
        key = tuple(zip(epi[i], sequence[epi[i]]))
        #then, we assign a random number to this interaction
        if key not in mem:
            if self._epi == 'uni':
                mem[key] = self._rng.uniform(0., 1.)
            elif self._epi == 'exp':
                mem[key] = self._rng.exponential(scale=1.)
            elif self._epi == 'zipf':
                mem[key] = float(self._rng.zipf(a=2.))
        return mem[key]

def registry() -> Dict[str, Dict]:
    problems = {
        "N4K0_standard": {
            "params": {
                "N": 4,
                "K": 0,
            },
            "starts": [
                "CDCC",
                "WDAM",
                "WSAC",
                "WDGM",
                "YDMQ",
            ],
        },
        "N4K0_exp": {
            "params": {
                "N": 4,
                "K": 0,
                "epi": 'exp',
            },
            "starts": [
                "WAHI",
                "FAHI",
                "WSHI",
                "HRGD",
                "HFGF",
                "WWIT",
                "CWAY",
                "CVGT",
                "RFSC",
                "HCWQ",
                "VPSM",
            ],
        },
        "N4K0_exp_zipf": {
            "params": {
                "N": 4,
                "K": 1,
                "epi": 'exp',
                "pos_weight": 'zipf',
            },
            "starts": [
                "DYYN",
                "DYAY",
                "DYEN",
                "DIYH",
                "VMYC",
                "TSCY",
                "NFEV",
                "TVMI",
                "EAMS",
                "RGMI",
                "YITT",
            ],
        },
        "N4K0_exp_zipf_hole0.3": {
            "params": {
                "N": 4,
                "K": 0,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "TQYY",
                "YMYY",
                "VDYH",
                "VDHY",
                "KDYN",
                "NDHV",
                "NDVC",
                "QDKA",
                "AQYT",
                "VDFW",
                "WCRV",
                "KLME",
                "SKQW",
                "WIQQ",
                "YLDI",
            ],
        },
        "N4K1_standard": {
            "params": {
                "N": 4,
                "K": 1,
            },
            "starts": [
                "EGWY",
                "EMWY",
                "ERWR",
                "KDWY",
                "EMQY",
                "EFLY",
                "EMWI",
                "ERQH",
                "SDAY",
                "TDSY",
                "EVYQ",
                "MFNY",
                "LASW",
                "AFNP",
                "SDDC",
                "MFAR",
                "WLNC",
            ],
        },
        "N4K1_exp": {
            "params": {
                "N": 4,
                "K": 1,
                "epi": 'exp',
            },
            "starts": [
                "SPCL",
                "SACL",
                "SVCL",
                "SWCL",
                "SNCL",
                "AACL",
                "KCCL",
                "SRCP",
                "NSCK",
                "ESCR",
                "WSRQ",
                "CICR",
                "MACH",
                "SRYA",
                "SWNI",
                "YQMQ",
                "YHKP",
                "RRYI",
                "GIFK",
                "TVDT",
            ],
        },
        "N4K1_exp_zipf": {
            "params": {
                "N": 4,
                "K": 1,
                "epi": 'exp',
                "pos_weight": 'zipf',
            },
            "starts": [
                "SLCW",
                "KPFM",
                "SHCY",
                "SPGD",
                "SCCV",
                "HPNQ",
                "SRQR",
                "IHCQ",
                "SQGL",
                "PPNT",
                "FGHV",
                "VMWW",
                "PIRP",
                "PNPH",
                "FNIP",
            ],
        },
        "N4K1_exp_zipf_hole0.3": {
            "params": {
                "N": 4,
                "K": 1,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "SPCW",
                "SPCR",
                "SPCG",
                "CPCM",
                "STCI",
                "LPCR",
                "SAPC",
                "MPCI",
                "IPCL",
                "MFYM",
                "SLTI",
                "KNTM",
                "MAGM",
                "RNTM",
                "NNVY",
                "HEDV",
                "MKVH",
                "WSMI",
                "YNFQ",
            ],
        },
        "N4K2_exp_zipf_hole0.3": {
            "params": {
                "N": 4,
                "K": 2,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "KQKS",
                "LQKS",
                "WNDS",
                "WVKC",
                "WQTH",
                "WQLD",
                "WQEP",
                "WECQ",
                "WALY",
                "YNKY",
                "QHSP",
                "MPKV",
                "SWFR",
                "TDIY",
                "MMNL",
                "QTHW",
                "FTYD",
            ],
        },
        "N4K3_exp_zipf_hole0.3": {
            "params": {
                "N": 4,
                "K": 3,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "RLSE",
                "RMDY",
                "RSCE",
                "RQVE",
                "FRLE",
                "SADE",
                "SSLE",
                "QSHE",
                "CLET",
                "FKDT",
                "LDTG",
                "SYVC",
                "CAFS",
                "SSTG",
                "TWWW",
                "CPQP",
            ],
        },
        "N5K1_exp_zipf_hole0.3": {
            "params": {
                "N": 5,
                "K": 1,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "LFTWW",
                "AFTWW",
                "DNTWW",
                "DLTWW",
                "NFTWW",
                "KETWW",
                "DPTWR",
                "CFGWW",
                "YPTWW",
                "IPTWW",
                "QYTRW",
                "QHTGW",
                "CDTQW",
                "MSTSW",
                "DSQWN",
                "GETLA",
                "SRSWN",
                "CFAKI",
                "GPYWY",
                "DWNFQ",
                "FWEFH",
                "YQAYA",
                "RWKFQ",
                "HKKTL",
                "HKCNH",
            ],
        },
        "N6K1_exp_zipf_hole0.3": {
            "params": {
                "N": 6,
                "K": 1,
                "epi": 'exp',
                "pos_weight": 'zipf',
                "cut_off": 0.3,
            },
            "starts": [
                "HPGSVV",
                "NPGSVH",
                "NPGSVR",
                "NPGSVN",
                "NPGSDR",
                "NPQSVN",
                "NPKSVC",
                "HPGSRV",
                "NPGQVR",
                "LPGQVP",
                "PPGSEH",
                "DPGHVC",
                "PPGGVF",
                "MPGDVR",
                "NTNVVF",
                "FMHSVC",
                "SPGRLF",
                "FPVSIP",
                "CRYSVN",
                "SPECSL",
                "DVQSYD",
                "AVGFMD",
                "RNAGHC",
                "LKPIST",
                "AFRVFY",
                "IHCKDD",
                "RFNDMC",
                "CYCKSN",
                "AGVWLE",
            ],
        },
    }
    return problems
