from typing import Tuple
from copy import deepcopy
from bisect import bisect_left

import random
import numpy as np
import pandas as pd

from fasthit.utils import sequence_utils as s_utils
from .acqfs import  UCB, LCB, TS, EI, PI, UCE, Greedy


class Enumerate(object):
    def __init__(self, explorer):
        self._explorer = explorer
        self._alphabet = explorer._alphabet
        self._seq_len = explorer._seq_len
        self._eval_batch_size = explorer._eval_batch_size
        self._candidates = self._enum_seqs()

    def _enum_seqs(self):
        new_seqs=[]
        def enum_seq(curr_seq):
            nonlocal new_seqs
            if len(curr_seq) == self._seq_len:
                new_seqs.append(curr_seq)
            else:
                for char in list(self._alphabet):
                    enum_seq(curr_seq + char)
        enum_seq('')
        return set(new_seqs)

    def sample(self, acqf, sampled_seqs, bsz=1):
        self._candidates.difference_update(sampled_seqs)
        candidates = list(self._candidates)

        maxima = []
        for seqs in np.array_split(candidates, len(candidates) / self._eval_batch_size):
            encodings = self._explorer.encoder.encode(seqs)
            preds = self._explorer.model.get_fitness(encodings)
            uncertainties = self._explorer.model.uncertainties
            acqf_val = acqf(preds, uncertainties, self._explorer._best_fitness)
            maxima.extend(
                [seqs[i], preds[i], acqf_val[i]]
                for i in range(len(seqs))
            )
        
        return sorted(maxima, reverse=True, key=lambda x: x[2])[:bsz]


class BoEvo(object):
    def __init__(self,explorer,add_random=True):
        self.explorer = explorer
        self.model = explorer._model
        self.alphabet=explorer._alphabet
        self._alphabet=explorer._alphabet
        self.seq_len = explorer._seq_len
        self.expmt_queries_per_round = explorer.expmt_queries_per_round
        self.model_queries_per_round = explorer.model_queries_per_round
        self._recomb_rate=0
        self._num_actions = 0
        self._seq_len = explorer._seq_len

    def _recombine_population(self, gen):
        np.random.shuffle(gen)
        ret = []
        for i in range(0, len(gen) - 1, 2):
            strA = []
            strB = []
            switch = False
            for ind in range(len(gen[i])):
                if np.random.random() < self._recomb_rate:
                    switch = not switch
                # putting together recombinants
                if switch:
                    strA.append(gen[i][ind])
                    strB.append(gen[i + 1][ind])
                else:
                    strB.append(gen[i][ind])
                    strA.append(gen[i + 1][ind])
            ret.append("".join(strA))
            ret.append("".join(strB))
        return ret
    def gen_random_seqs(self,n,acqf):
        seq_array = np.random.randint(len(self.alphabet),size=(n,self.seq_len))
        seq_list = [ ''.join([ self.alphabet[j] for j in seq_array[i]]) for i in range(n)] 
        encodings = self.explorer.encoder.encode(seq_list)
        Mu, std = self.explorer.model._predict(encodings)
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return [ {'seq':seq_list[i],'fitness':fval[i]} for i in range(len(seq_list)) ]

    def acquire(self,acqf,all_measured_seqs,K=384,plot=False):#,cycles=1000,population_size=100,sample_size=10
        self.fval_s=[]
        starting_sequence = self.gen_random_seqs(1,acqf)[0]
        self._state = s_utils.string_to_one_hot(starting_sequence['seq'], self.alphabet)
        self.population = pd.DataFrame(
            {
                'sequence':starting_sequence['seq'],
                'fval':[starting_sequence['fitness']],
                'round':0,
                'model_cost':self.explorer._model.cost,
            }
        )
        for r in range(8):
            std,seqs,fvals=self.optimize(self.population,acqf)
            self.population = self.population.append(
                pd.DataFrame(
                    {
                        'sequence':seqs,
                        'fval':fvals,
                        'round':r,
                        'model_cost':self.explorer._model.cost
                    }
                )
            )
        # self.population['fval'].scatter(marker=".")
        self.fval_s=np.array(self.fval_s)
        if self.fval_s.max()==self.fval_s.min():
            print(" boevo opt does not improve at all")
            print(self.fval_s.max())
        if plot:
            plt.scatter(
                range(len(self.fval_s)),
                self.fval_s,
                marker='.'
            )
            # plt.title(str(cur_batch))
            plt.show()
        his_seqs=self.population['sequence'].values
        his_fvals = self.population['fval'].values
        sort_idx = np.argsort(his_fvals)[::-1]
        result_idx=[]
        for idx in sort_idx:
            if len(result_idx)>=K:
                break
            elif his_seqs[idx] not in all_measured_seqs:
                result_idx.append(idx)
        assert len(result_idx)==K
        
        return [{'fitness':his_fvals[idx],'seq':his_seqs[idx]} for idx in result_idx]
        # return his_seqs[idx],np.zeros_like(his_seqs)
    
    @staticmethod
    def Thompson_sample(measured_batch):
        """Pick a sequence via Thompson sampling."""
        fitnesses = np.cumsum([np.exp(10 * x[0]) for x in measured_batch])
        fitnesses = fitnesses / fitnesses[-1]
        x = np.random.uniform()
        index = bisect_left(fitnesses, x)
        sequences = [x[1] for x in measured_batch]
        return sequences[index]
    def get_fval(self,acqf,seq):
        encodings = self.explorer.encoder.encode(seq)
        Mu, std = self.model._predict(encodings)
        fval = acqf(Mu,std,self.explorer._best_fitness)
        self.fval_s=self.fval_s+list(fval)
        return fval,std

    def optimize(
        self,
        measured_sequences: pd.DataFrame,
        acqf,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Propose top `expmt_queries_per_round` sequences for evaluation."""
        if self._num_actions > 0:
            # set state to best measured sequence from prior batch
            last_round_num = measured_sequences["round"].max()
            last_batch = measured_sequences[
                measured_sequences["round"] == last_round_num
            ]
            _last_batch_seqs = last_batch["sequence"].tolist()
            _last_batch_true_scores = last_batch["fval"].tolist()
            last_batch_seqs = _last_batch_seqs
            if self._recomb_rate > 0 and len(last_batch) > 1:
                last_batch_seqs = self._recombine_population(last_batch_seqs)
            measured_batch = []
            _new_seqs = []
            for seq in last_batch_seqs:
                if seq in _last_batch_seqs:
                    measured_batch.append(
                        (_last_batch_true_scores[_last_batch_seqs.index(seq)], seq)
                    )
                else:
                    _new_seqs.append(seq)
            if len(_new_seqs) > 0:
                fitnesses,std = self.get_fval(acqf,_new_seqs)
                measured_batch.extend(
                    [(fitnesses[i], _new_seqs[i]) for i in range(fitnesses.shape[0])]
                )
            measured_batch = sorted(measured_batch)
            sampled_seq = self.Thompson_sample(measured_batch)
            self._state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
        # generate next batch by picking actions
        initial_uncertainty = None
        samples = []
        preds = []
        prev_cost = self.model.cost
        all_measured_seqs = set(measured_sequences["sequence"].tolist())
        zzz=0
        while self.model.cost - prev_cost < self.model_queries_per_round:
            zzz+=1
            uncertainty, new_state_string, pred = self.pick_action(acqf)
            if new_state_string not in all_measured_seqs:
                # self._best_fitness = max(self._best_fitness, pred)
                all_measured_seqs.add(new_state_string)
                samples.append(new_state_string)
                preds.append(pred)
            if initial_uncertainty is None:
                initial_uncertainty = uncertainty
            if uncertainty > 2. * initial_uncertainty:
                # reset sequence to starting sequence if we're in territory that's too
                # uncharted
                sampled_seq = self.Thompson_sample(measured_batch)
                self._state = s_utils.string_to_one_hot(sampled_seq, self._alphabet)
                initial_uncertainty = None
        if len(samples) < self.expmt_queries_per_round:
            random_sequences = set()
            while len(random_sequences) < self.expmt_queries_per_round - len(samples):
                # TODO: redundant random samples
                random_sequences.update(
                    s_utils.generate_random_sequences(
                        self._seq_len,
                        self.expmt_queries_per_round - len(samples) - len(random_sequences),
                        list(self._alphabet)
                    )
                )
            random_sequences = sorted(random_sequences)
            samples.extend(random_sequences)
            # encodings = self.explorer.encoder.encode(random_sequences)
            preds.extend(self.get_fval(acqf,random_sequences)[0])
        samples = np.array(samples)
        preds = np.array(preds)
        return measured_sequences, samples, preds
        # sorted_order = np.argsort(preds)[: -self.expmt_queries_per_round-1 : -1]
        # return measured_sequences, samples[sorted_order], preds[sorted_order]

    def sample_actions(self):
        """Sample actions resulting in sequences to screen."""
        actions = set()
        pos_changes = []
        for pos in range(self._seq_len):
            pos_changes.append([])
            for res in range(len(self._alphabet)):
                if self._state[pos, res] == 0:
                    pos_changes[pos].append((pos, res))
        while len(actions) < self.model_queries_per_round / self.expmt_queries_per_round:
            action = []
            for pos in range(self._seq_len):
                if np.random.random() < 1. / self._seq_len:
                    pos_tuple = pos_changes[pos][
                        np.random.randint(len(self._alphabet) - 1)
                    ]
                    action.append(pos_tuple)
            if len(action) > 0 and tuple(action) not in actions:
                actions.add(tuple(action))
        return list(actions)

    def pick_action(self,acqf):
        """Pick action."""
        ### select 1 from n candidates
        state = self._state.copy()
        actions = self.sample_actions()
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.model_queries_per_round // self.expmt_queries_per_round):
            x = np.zeros((self._seq_len, len(self._alphabet)))
            for action in actions[i]:
                x[action] = 1
            actions_to_screen.append(x)
            state_to_screen = s_utils.construct_mutant_from_sample(x, state)
            states_to_screen.append(s_utils.one_hot_to_string(state_to_screen, self._alphabet))
        encodings = self.explorer.encoder.encode(states_to_screen)


        preds,std = self.get_fval(acqf,states_to_screen)
        action_idx = np.argmax(preds) 
        uncertainty = std[action_idx]
        action = actions_to_screen[action_idx]
        new_state_string = states_to_screen[action_idx]
        self._state = s_utils.string_to_one_hot(new_state_string, self._alphabet)
        self._num_actions += 1
        return uncertainty, new_state_string, preds[action_idx]


class AgeEvo(object):
    def __init__(self,explorer,add_random=False):
        self.explorer = explorer
        self.alphabet=explorer._alphabet
        self.seq_len = explorer._seq_len
        self.add_random=add_random
        # self.acqf = explorer.acqf
    def gen_random_seqs(self,n,acqf):
        seq_array = np.random.randint(len(self.alphabet),size=(n,self.seq_len))
        seq_list = [ ''.join([ self.alphabet[j] for j in seq_array[i]]) for i in range(n)] 
        encodings = self.explorer.encoder.encode(seq_list)
        Mu, std = self.explorer.model._predict(encodings)
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return [ {'seq':seq_list[i],'fitness':fval[i]} for i in range(len(seq_list)) ]
    def mutate(self,parent,acqf):
        seq = parent['seq']
        mutate_point = np.random.randint(len(seq))
        seq=seq[:mutate_point] +np.random.choice(list(self.alphabet)) +seq[mutate_point+1:]

        encodings = self.explorer.encoder.encode([seq])
        Mu, std = self.explorer.model._predict(encodings)
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return {'seq':seq,'fitness':fval}
    def init_seqs(self,n,acqf,all_measured_seqs):
        seq_list = all_measured_seqs[-n:]
        if len(seq_list)==0:
            seq_array = np.random.randint(len(self.alphabet),size=(n-len(seq_list),self.seq_len))
            seq_list = seq_list+ [ ''.join([ self.alphabet[j] for j in seq_array[i]]) for i in range(n)] 
        # assert len(seq_list)==n
        encodings = self.explorer.encoder.encode(seq_list)
        Mu, std = self.explorer.model._predict(encodings)
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return [ {'seq':seq_list[i],'fitness':fval[i]} for i in range(len(seq_list)) ]

    def acquire(self,acqf,all_measured_seqs,K=384,plot=False):
        
        cycles=3000
        population_size=768
        sample_size=10
        all_measured_seqs=list(all_measured_seqs)
        history = []
        population = self.init_seqs(population_size,acqf=acqf,all_measured_seqs=all_measured_seqs)# list of dict
        if self.add_random:
            population = random.choices(population,k=population_size//2)
            random_pop = self.gen_random_seqs(population_size-len(population),acqf)
            population = random_pop+population
            np.random.shuffle(population)
            history = deepcopy(population)
        best_fval=-np.inf
        not_improve = 0
        while True:
            sample = random.choices(population,k=sample_size)
            parent = max(sample, key=lambda x: x['fitness'])
            child = self.mutate(parent,acqf=acqf)
            population.append(child)
            history.append(child)
            if len(population)>population_size:
                population=population[-population_size:]
            not_improve +=1
            if child['fitness']>best_fval:
                best_fval=child['fitness']
                not_improve =0
            elif not_improve>cycles:
                break
        if plot:
            plt.scatter(range(len(history)),[i['fitness'] for i in history],marker=".",linewidths =0)
            plt.show()
        his_fvals = np.array([i['fitness'] for i in history]).flatten()
        his_seqs = np.array([i['seq'] for i in history]).flatten()
        sort_idx = np.argsort(his_fvals)[::-1]
        result_idx=[]
        for idx in sort_idx:
            if len(result_idx)>=K:
                break
            elif his_seqs[idx] not in all_measured_seqs:
                result_idx.append(idx)
        assert len(result_idx)==K
        # samples = [history[idx]['seq'] for idx in result_idx]
        # print(result_idx)
        return [{'fitness':his_fvals[idx],'seq':his_seqs[idx]} for idx in result_idx]


class NewEvo(object):
    def __init__(self, explorer):
        self.explorer = explorer
        self.alphabet = explorer._alphabet
        self.seq_len = explorer._seq_len
        self.add_random = True

    def _gen_random_seqs(self, n, acqf):
        seq_array = np.random.randint(len(self.alphabet),size=(n,self.seq_len))
        seq_list = [ ''.join([ self.alphabet[j] for j in seq_array[i]]) for i in range(n)] 
        encodings = self.explorer.encoder.encode(seq_list)
        #Mu, std = self.explorer.model._predict(encodings)
        Mu = self.explorer.model.get_fitness(encodings)
        std = self.explorer.model.uncertainties
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return [ {'seq':seq_list[i],'fitness':fval[i]} for i in range(len(seq_list)) ]
    
    def _mutate(self, parent):
        result = []
        for seq in parent:
            mutate_point = np.random.randint(len(seq))
            seq = seq[:mutate_point] + np.random.choice(list(self.alphabet)) + seq[mutate_point + 1 :]
            result.append(seq)
        return result
    
    def _crossover(self, parent):
        assert len(parent) == 2
        mom, dad = parent
        child = ""
        for i in range(len(mom)):
            child += mom[i] if np.random.rand() > 0.5 else dad[i]
        return child

    def evaluate(self, seqs, acqf):
        encodings = self.explorer.encoder.encode(seqs)
        #Mu, std = self.explorer.model._predict(encodings)
        Mu = self.explorer.model.get_fitness(encodings)
        std = self.explorer.model.uncertainties
        fval = acqf(Mu, std, self.explorer._best_fitness)
        return [ {'seq':seqs[i],'fitness':fval[i]}  for i in range(len(seqs))]
    
    def _init_seqs(self,n,acqf,all_measured_seqs):
        seq_list = all_measured_seqs[-n:]
        if len(seq_list)==0:
            seq_array = np.random.randint(len(self.alphabet),size=(n-len(seq_list),self.seq_len))
            seq_list = seq_list+ [ ''.join([ self.alphabet[j] for j in seq_array[i]]) for i in range(n)] 
        # assert len(seq_list)==n
        encodings = self.explorer.encoder.encode(seq_list)
        #Mu, std = self.explorer.model.get_fitness(encodings)
        Mu = self.explorer.model.get_fitness(encodings)
        std = self.explorer.model.uncertainties
        fval = acqf(Mu,std,self.explorer._best_fitness)
        return [ {'seq':seq_list[i],'fitness':fval[i]} for i in range(len(seq_list)) ]

    def sample(self, acqf, sampled_seqs, bsz=1):
        
        cycles=200
        population_size=768
        sample_size=10
        all_measured_seqs=list(sampled_seqs)
        population = self._init_seqs(
            population_size,
            acqf=acqf,
            all_measured_seqs=all_measured_seqs
        )# list of dict
        history = deepcopy(population)
        if self.add_random:
            population = random.choices(population, k=population_size // 2)
            random_pop = self._gen_random_seqs(population_size - len(population), acqf)
            population += random_pop
            np.random.shuffle(population)
            history += random_pop
        best_fval = -np.inf
        not_improve = 0
        while True:
            offsprings = []
            to_mutate=[]
            for _ in range(10):
                mom = max(random.choices(population, k=sample_size), key=lambda x: x['fitness'])['seq']
                dad = max(random.choices(population, k=sample_size), key=lambda x: x['fitness'])['seq']
                to_mutate += [mom, dad]
                offsprings.append(self._crossover([mom, dad]))
            to_mutate = to_mutate + offsprings
            offsprings += self._mutate(to_mutate)
            offsprings = self.evaluate(offsprings,acqf)

            population += offsprings
            history += offsprings
            if len(population) > population_size:
                population = population[-population_size:]
            not_improve += 1
            max_fitness_in_offspring = max(offsprings, key=lambda x:x['fitness'])['fitness']
            if max_fitness_in_offspring > best_fval:
                best_fval = max_fitness_in_offspring
                not_improve = 0
            elif not_improve > cycles:
                break
        
        his_fvals = np.array([i['fitness'] for i in history]).flatten()
        his_seqs = np.array([i['seq'] for i in history]).flatten()
        sort_idx = np.argsort(his_fvals)[::-1]
        result_idx=[]
        for idx in sort_idx:
            if his_seqs[idx] not in all_measured_seqs:
                result_idx.append(idx)
        
        sorted_samples = [[his_seqs[idx], his_fvals[idx]] for idx in result_idx]

        return sorted_samples[:bsz]
