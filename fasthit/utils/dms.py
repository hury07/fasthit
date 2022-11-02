import numpy as np
import pandas as pd
import fasthit.utils.sequence_utils as s_utils

def generate_dms_variants(seq, alphabet):
    alphabet = set(alphabet)
    assert all([aa in alphabet for aa in seq])
    ###
    seq_length = len(seq)
    alphabet_size = len(alphabet)
    n_seqs = seq_length * (alphabet_size - 1)
    ###
    seqs = [None] * n_seqs
    for i in range(seq_length):
        vocab = alphabet.difference(seq[i])
        for j, aa in enumerate(vocab):
            if aa != seq[i]:
                temp = list(seq)
                temp[i] = aa
                seqs[i * len(vocab) + j] = ''.join(temp)
    return np.array(seqs)


def get_dms(landscape, seq, path=None):
    seqs = generate_dms_variants(seq, s_utils.AAS)
    scores = landscape.get_fitness(seqs)
    seqs = pd.DataFrame(
        {
            "Variants": seqs,
            "Fitness": scores,
        }
    )
    if path is not None:
        seqs.to_csv(path, index=False)
    return seqs
