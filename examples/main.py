import os
import sys
import pandas as pd
import toml
import itertools
import time
import torch

import fasthit
from fasthit.utils import utils
import fasthit.utils.sequence_utils as s_utils
from fasthit.utils.dms import generate_dms_variants


def make_landscape(type_name, spec_name):
    if type_name == "gb1":
        problem = fasthit.landscapes.gb1.registry()[spec_name]
        landscape = fasthit.landscapes.GB1(**problem["params"])
        alphabet = s_utils.AAS
    elif type_name == "phoq":
        problem = fasthit.landscapes.phoq.registry()[spec_name]
        landscape = fasthit.landscapes.PhoQ(**problem["params"])
        alphabet = s_utils.AAS
    elif type_name == "nk":
        problem = fasthit.landscapes.nk.registry()[spec_name]
        landscape = fasthit.landscapes.NK(**problem["params"])
        alphabet = s_utils.AAS
    elif type_name == "exp":
        problem = fasthit.landscapes.exp.registry()[spec_name]
        landscape = fasthit.landscapes.EXP(**problem["params"])
        alphabet = s_utils.AAS
    else:
        pass
    return problem, landscape, alphabet

def make_encoder(name, landscape, alphabet):
    if type(name) is list:
        encoder = fasthit.encoders.Fusion(
            alphabet,
            name,
            landscape=landscape,
            method='concat',
        )
    elif name == "onehot":
        encoder = fasthit.encoders.OneHot(alphabet)
    elif name == "georgiev":
        encoder = fasthit.encoders.Georgiev(alphabet)
    elif name in ["transformer", "unirep", "trrosetta"]:
        encoder = fasthit.encoders.TAPE(
            name, alphabet,
            landscape.wt, landscape.combo_python_idxs,
        )
    elif name in ["esm-1b", "esm-1v", "esm-msa-1", "esm-msa-1b"]:
        encoder = fasthit.encoders.ESM(
            name, alphabet,
            landscape.wt, landscape.combo_python_idxs,
            batch_size=128,
        )
    elif name in ["prot_bert_bfd", "prot_t5_xl_uniref50"]:
        encoder = fasthit.encoders.ProtTrans(
            name, alphabet,
            landscape.wt, landscape.combo_python_idxs,
            batch_size=128,
        )
    elif name in ["esm-tok"]:
        encoder = fasthit.encoders.ESM_Tokenizer(
            "esm-1v",
            landscape.wt, landscape.combo_python_idxs,
            pretrained_model_dir="/home/hury/src/fasthit/pretrained_models/esm/",
        )
    else:
        pass
    return encoder

def make_model(name, seq_length, n_features, landscape, **kwargs):
    if name == "linear":
        model = fasthit.models.LinearRegression()
    elif name == "randomforest":
        model = fasthit.models.RandomForestRegressor()
    elif name == "mlp":
        model = fasthit.models.MLP(
            seq_length=seq_length, input_features=n_features,
        )
    elif name == "cnn":
        model = fasthit.models.CNN(
            n_features, num_filters=32, hidden_size=64, kernel_size=2,
        )
    elif name == "ensemble":
        model = fasthit.models.Ensemble(
            [
                fasthit.models.MLP(
                    seq_length=seq_length, input_features=n_features,
                ) for _ in range(3)
            ]
        )
    elif name == "gpr":
        model = fasthit.models.GPRegressor(kernel=kwargs["kernel"])
    elif name == "rio":
        model = fasthit.models.RIO(
            fasthit.models.MLP(
                seq_length=seq_length, input_features=n_features,
            ),
        )
    elif name == "finetune":
        model = fasthit.models.Finetune(
            pretrained_model_dir="/home/hury/src/fasthit/pretrained_models/esm/",
            pretrained_model_name="esm-1v",
            pretrained_model_exprs=[33],
            target_python_idxs=landscape.combo_python_idxs,
        )
    else:
        pass
    return model

def make_explorer(
    name, alphabet, encoder, model,
    start_seq, rounds, bz_expmt, bz_model,
    log_file,
    **kwargs
):
    if name == "random":
        explorer = fasthit.explorers.Random(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            seed=kwargs["seed"],
        )
    elif name == "random_ms":
        explorer = fasthit.explorers.Random(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            seed=kwargs["seed"],
            elitist=True,
        )
    elif name == "adalead":
        explorer = fasthit.explorers.Adalead(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            seed=kwargs["seed"],
        )
    elif name == "bo_enu":
        explorer = fasthit.explorers.bo.BO_ENU(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            seed=kwargs["seed"],
            util_func=kwargs["util_func"],
            uf_param=kwargs["uf_param"],
        )
    elif name == "bo_evo":
        explorer = fasthit.explorers.bo.BO_EVO(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            seed=kwargs["seed"],
            util_func=kwargs["util_func"],
            uf_param=kwargs["uf_param"],
        )
    else:
        pass
    return explorer

def main(config_file):
    directory = os.path.split(config_file)[-1].split('.')[1]
    cfg = toml.load(config_file)
    ###
    eval_models = cfg.pop('eval_mode')
    testset_path = cfg.pop('testset_path')
    testset = pd.read_csv(testset_path).to_numpy()
    ###
    output_dir = cfg.pop('output_dir')
    verbose = cfg.pop('verbose')
    starts = cfg.pop('starts')
    seeds = cfg.pop('seeds')
    ###
    elapsed_times = []
    for cur_iter in itertools.product(*cfg.values()):
        landscape_name, \
        explorer_name, bo_util_func, bo_uf_param, \
        expmt_queries_per_round, model_queries_per_round, \
        encoding, model_name, gp_kernel, \
        warm_start, rounds \
        = cur_iter
        
        ###
        type_name, spec_name = landscape_name.split(":")
        problem, landscape, alphabet = make_landscape(type_name, spec_name)
        encoder = make_encoder(encoding, landscape, alphabet)
        ###
        for i, seed in enumerate(seeds):
            kwargs = {
                "seed": seed,
                "util_func": bo_util_func,
                "uf_param": bo_uf_param,
                "kernel": gp_kernel,
            }
            for j, start_seq in enumerate(problem[starts]):
                utils.set_torch_seed(seed)
                ###
                model = make_model(
                    model_name, len(start_seq), encoder.n_features, landscape,
                    **kwargs
                )
                ###
                k = i*len(problem[starts]) + j
                subdirs = {
                    'rounds': expmt_queries_per_round,
                    'budget': expmt_queries_per_round,
                    'model': encoding,
                    'encoder': encoding,
                    'warm_start': warm_start,
                    'explorer': explorer_name,
                    'landscape': spec_name,
                    'MER': int(model_queries_per_round/expmt_queries_per_round),
                    'wt_only': warm_start,
                }
                log_file = (
                    f"{output_dir}/{type_name}/"
                    + f"{directory}/"
                    + f"{subdirs[directory]}"
                    + f"/run{k}.csv"
                )
                eval_file = (
                    f"{output_dir}/{type_name}/"
                    + f"{directory}/"
                    + f"{subdirs[directory]}"
                    + f"/eval{k}.csv"
                )
                ###
                explorer = make_explorer(
                    explorer_name, alphabet, encoder, model, start_seq, rounds,
                    expmt_queries_per_round, model_queries_per_round, log_file,
                    **kwargs
                )
                start = time.time()
                if warm_start == 'warm':
                    explorer.run(
                        landscape, verbose=False,
                        init_seqs=generate_dms_variants(start_seq, alphabet)
                    )
                else:
                    explorer.run(
                        landscape, verbose=False,
                        eval_models=eval_models, testset=testset, eval_file=eval_file,
                    )
                elapsed_times.append([spec_name, explorer_name, time.time() - start, k])
                del explorer, model
                torch.cuda.empty_cache()
    if verbose:
        elapsed_times = pd.DataFrame(
            elapsed_times,
            columns=['landscape', 'explorer', 'time(s)', 'simulation']
        )
        elapsed_times.to_csv(f"{output_dir}/{type_name}/{directory}/time.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1])
