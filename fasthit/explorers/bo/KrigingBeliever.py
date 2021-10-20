from typing import Optional, Tuple
import numpy as np
import pandas as pd

import fasthit
from fasthit.utils import sequence_utils as s_utils

from .acqfs import  UCB,LCB,TS,EI,PI,UCE,Greedy
from .acqf_optimizers import BoEvo, Enumerate, AgeEvo, NewEvo


class BO_KB(fasthit.Explorer):

    def __init__(
        self,
        encoder: fasthit.Encoder,
        model: fasthit.Model,
        rounds: int,
        expmt_queries_per_round: int, # default: 384
        model_queries_per_round: int, # default: 800
        starting_sequence: str,
        alphabet: str = s_utils.AAS,
        log_file: Optional[str] = None,
        acqf: str = "LCB",
        eval_batch_size: int = 256,
        callback_func: str = "orig_kb",
        acqf_optim: str = "boevo",
        kb_in_first_batch = True, # if False, the first batch will be acquired by simple batch method
        acqf_count_per_iter=200,
    ):
        """Initialize the explorer."""
        name = f"BO_KB_Explorer-acqftion={acqf}"
        assert hasattr(model, "uncertainties")
        ###
        super().__init__(
            name,
            encoder,
            model,
            rounds,
            expmt_queries_per_round,
            model_queries_per_round,
            starting_sequence,
            log_file,
        )
        self._alphabet = alphabet
        self._best_fitness = 0.
        self._seq_len = len(starting_sequence)
        self._eval_batch_size = eval_batch_size
        acqfs = {
            "UCB": UCB,
            "LCB": LCB,
            "TS": TS,
            "EI": EI,
            "PI": PI,
            "Greedy": Greedy,
        }
        self._acqf = acqfs[acqf]

        callback_funcs = {
            "random": self.random_cb,
            "random_inv": self.adapt_random_cb_inverse,
            "seq": self.sequential_cb,
            "seq_inv": self.sequential_cb_reverse,
            "adapt": self.adapt_random_cb,
            "adapt_inv": self.adapt_random_cb_inverse,
            "orig_kb": self.orig_kb,
        }
        
        self.callback_func=callback_funcs[callback_func]
        
        acqf_optims = {
            'boevo': BoEvo,
            'enum': Enumerate,
            'agevo': AgeEvo,
            'newevo': NewEvo,
        }
        self.acqf_optim = acqf_optims[acqf_optim](self)
        self.kb_in_first_batch = kb_in_first_batch

    def _pick_seq(self, all_measured_seqs, use_uncertainty=False):
        acqf = UCE if use_uncertainty else self._acqf

        samples = self.acqf_optim.sample(
            acqf,
            sampled_seqs=all_measured_seqs,
            bsz=1,
        )
        return samples[0][:2]

    def gen_random_seqs(self, n, all_measured_seqs):
        seq_array = np.random.randint(len(self._alphabet), size = (2*n, self._seq_len))
        seq_list = [''.join([self._alphabet[j] for j in seq_array[i]]) for i in range(2 * n)]
        seq_list = [i for i in seq_list if i not in all_measured_seqs]
        
        return seq_list[:n]

    def propose_first_batch(
        self,
        measured_sequences: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose `batch_size` samples using Krigging Believer"""
        samples = []
        preds = []

        all_measured_seqs = set(measured_sequences["sequence"].values)
        self._best_fitness = measured_sequences['true_score'].max()
        
        history = []
        history = history + self.acqf_optim.sample(
            self._acqf,
            all_measured_seqs = all_measured_seqs,
            K = self.expmt_queries_per_round,
        )
        # history = set(history)
        print('length of history:', len(history))
        history = sorted(history, reverse = True, key = lambda x: x['fitness'])
        for child in history:
            if child['seq'] not in all_measured_seqs:
                samples.append(child['seq'])
                preds.append(0)
            if len(samples) == self.expmt_queries_per_round:
                break
        return measured_sequences, np.array(samples), np.array(preds)

    def propose_sequences(
        self,
        measured_sequences: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        last_round = measured_sequences["round"].max()
        # adapt simple_batch_method to acquire the first batch
        if (
            not self.kb_in_first_batch
            and last_round == 0
        ):
            print('BOEVO_simple_batch_method to generate the first batch')
            return self.propose_first_batch(measured_sequences)
        
        """Propose `batch_size` samples using Krigging Believer"""
        samples = []
        preds = []
        all_measured_seqs = set(measured_sequences["sequence"].values)
        
        self._best_fitness = measured_sequences['true_score'].max()
        encodings = self.encoder.encode(measured_sequences["sequence"].to_list())
        labels = measured_sequences["true_score"].to_numpy()

        import tqdm
        for i in  tqdm.trange(self.expmt_queries_per_round):
            # Acquire the best seq from sequential mode acqf
            use_std = self.callback_func(i, self.expmt_queries_per_round)
            new_seq, new_pred = self._pick_seq(all_measured_seqs, use_std)
            self._best_fitness = max(self._best_fitness, new_pred)
            
            samples.append(new_seq)
            all_measured_seqs.add(new_seq)
            preds.append(new_pred)

            # extend training data with imagined label from model
            new_encoding = self.encoder.encode([new_seq])
            new_label = np.asarray([new_pred])
            encodings = np.concatenate([encodings, new_encoding], axis=0)
            labels = np.concatenate([labels, new_label], axis=0)

            self.model.train(encodings, labels)

        return measured_sequences, np.array(samples), np.array(preds)


    def get_training_data(
        self,
        measured_sequences: pd.DataFrame,
    ) -> pd.DataFrame:
        return measured_sequences


    def random_cb(self, cur_q, batch_size):
        if cur_q == 0:
            return False
        if np.random.rand() > 0.1:
            return False
        return True

    def sequential_cb(self, cur_q, batch_size):
        if cur_q == 0:
            return False
        if cur_q < 0.1*batch_size:
            return True
        return False

    def sequential_cb_reverse(self, cur_q, batch_size):
        if cur_q == 0:
            return False
        if cur_q < 0.1*batch_size:
            return False
        return True

    def adapt_random_cb(self, cur_q, batch_size):
        if cur_q == 0:
            return False
        if np.random.rand() > (cur_q + 1) / batch_size:
            return True
        return False

    def adapt_random_cb_inverse(self, cur_q, batch_size):
        if cur_q == 0:
            return False
        if np.random.rand() > (cur_q + 1) / batch_size:
            return False
        return True

    def orig_kb(self, cur_q, batch_size):
        return False
