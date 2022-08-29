import itertools
import torch

import fasthit
from fasthit.utils import utils
import fasthit.utils.sequence_utils as s_utils


def make_landscape(type_name, spec_name):
    if type_name == "tf-binding":
        problem = fasthit.landscapes.tf_binding.registry()[spec_name]
        landscape = fasthit.landscapes.TFBinding(**problem["params"])
        alphabet = s_utils.DNAA
    elif type_name == "rna":
        problem = fasthit.landscapes.rna.registry()[spec_name]
        landscape = fasthit.landscapes.RNABinding(**problem["params"])
        alphabet = s_utils.RNAA
    elif type_name == "rosetta":
        problem = fasthit.landscapes.rosetta.registry()[spec_name]
        landscape = fasthit.landscapes.RosettaFolding(**problem["params"])
        alphabet = s_utils.AAS
    elif type_name == "gb1":
        problem = fasthit.landscapes.gb1.registry()[spec_name]
        landscape = fasthit.landscapes.GB1(**problem["params"])
        alphabet = s_utils.AAS
    elif type_name == "exp":
        problem = fasthit.landscapes.exp.registry()[spec_name]
        landscape = fasthit.landscapes.EXP(**problem["params"])
        alphabet = s_utils.AAS
    else:
        pass
    return problem, landscape, alphabet

def make_encoder(name, landscape, alphabet):
    if name == "onehot":
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
        )
    elif name in ["prot_bert_bfd", "prot_t5_xl_uniref50"]:
        encoder = fasthit.encoders.ProtTrans(
            name, alphabet,
            landscape.wt, landscape.combo_python_idxs,
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

def main():
    seeds = [0]#[0, 12, 123, 1234, 12345]
    rounds = 10
    expmt_queries_per_round = 384
    model_queries_per_round = 3200

    landscape_names = ["gb1:with_imputed"]
    # "rna:L14_RNA1", "rna:L50_RNA1", "rna:L100_RNA1", "rna:C20_L100_RNA1+2",
    # "rosetta:3msi", "gb1:with_imputed"
    encodings = ["esm-1v"]
    # "onehot",
    # "georgiev",
    # "transformer", "unirep", "trrosetta",
    # "esm-1b", "esm-1v", "esm-msa-1", "esm-msa-1b", "esm-tok",
    # "prot_bert_bfd", "prot_t5_xl_uniref50",
    model_names = ["gpr"]
    # "linear", "randomforest",
    # "mlp", "cnn",
    # "gpr", "rio",
    # "ensemble",
    # "finetune"
    gp_kernels = ["RBF"]
    explorer_names = ["bo_evo"]
    # "random", "adalead", "bo_evo", "bo_enu",
    # "mlde", "rl"
    bo_util_funcs = ["UCB"]
    # "UCB", "LCB", "EI", "PI", "TS"
    bo_uf_params = [0.2]
    

    for cur_iter in itertools.product(
        landscape_names,
        encodings,
        model_names,
        gp_kernels,
        explorer_names,
        bo_util_funcs,
        bo_uf_params,
    ):
        landscape_name, encoding, model_name, gp_kernel, explorer_name, \
        bo_util_func, bo_uf_param\
        = cur_iter
        ###
        
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
            for j, start_seq in enumerate(problem["starts"]):
                utils.set_torch_seed(seed)
                ###
                model = make_model(
                    model_name, len(start_seq), encoder.n_features, landscape,
                    **kwargs
                )

                k = i * len(problem["starts"]) + j
                log_file = (
                    f"runs_{type_name}/{spec_name}/"
                    + f"{explorer_name}/{encoding}/{model_name}/"
                    + f"test"
                    + f"/run{k}.csv"
                )
                ###
                explorer = make_explorer(
                    explorer_name, alphabet, encoder, model, start_seq, rounds,
                    expmt_queries_per_round, model_queries_per_round, log_file,
                    **kwargs
                )
                explorer.run(landscape, verbose=False)#, init_seqs_file="data/gb1/init_seqs.csv")
                del explorer, model
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
