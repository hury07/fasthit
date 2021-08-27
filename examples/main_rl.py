import numpy as np
import torch

import fasthit
from fasthit.utils import utils
import fasthit.utils.sequence_utils as s_utils
from config_rl import cfg


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
    else:
        pass
    return encoder

def make_model(name, seq_length, n_features, **kwargs):
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
        )
    elif name == "mlde":
        explorer = fasthit.explorers.MLDE(
            encoder,
            model,
            rounds=4,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
        )
    elif name == "bo_enu":
        explorer = fasthit.explorers.BO_ENU(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            proposal_func=kwargs["ac_func"],
        )
    elif name == "bo_evo":
        explorer = fasthit.explorers.BO_EVO(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            alphabet=alphabet,
            log_file=log_file,
            proposal_func=kwargs["ac_func"],
        )
    elif name == "ppo_m":
        explorer = fasthit.explorers.RL(
            encoder,
            model,
            rounds=rounds,
            expmt_queries_per_round=bz_expmt,
            model_queries_per_round=bz_model,
            starting_sequence=start_seq,
            cfg = kwargs["cfg"],
            alphabet=alphabet,
            log_file=log_file,
        )
    else:
        pass
    return explorer

def main():
    seed = 42
    rounds = 10
    expmt_queries_per_round = 384
    model_queries_per_round = 3200
    ###
    for landscape_name in ["gb1:with_imputed"]:
        # "rna:L14_RNA1", "rna:L50_RNA1", "rna:L100_RNA1", "rna:C20_L100_RNA1+2",
        # "rosetta:3msi", "gb1:with_imputed"
        type_name, spec_name = landscape_name.split(":")
        problem, landscape, alphabet = make_landscape(type_name, spec_name)
        kwargs = {
            "ac_func": "LCB",
            "kernel": "RBF",
            "cfg": cfg,
        }
        for encoding in ["onehot"]:
            # "onehot",
            # "georgiev",
            # "transformer", "unirep", "trrosetta",
            # "esm-1b", "esm-1v", "esm-msa-1", "esm-msa-1b"
            # "prot_bert_bfd", "prot_t5_xl_uniref50",
            encoder = make_encoder(encoding, landscape, alphabet)
            for explorer_name in ["ppo_m"]:
                # "random", "adalead", "bo_evo", "bo_enu",
                # "mlde"
                for model_name in ["mlp"]:
                    # "linear", "randomforest"
                    # "mlp", "cnn",
                    # "gpr", "rio",
                    # "ensemble"
                    for i, start_seq in enumerate(problem["starts"]):
                        np.random.seed(seed)
                        utils.set_torch_seed(seed)
                        ###
                        model = make_model(
                            model_name, len(start_seq), encoder.n_features,
                            **kwargs
                        )
                        log_file = (
                            f"runs_{type_name}/{spec_name}/"
                            + f"{explorer_name}/{encoding}/{model_name}/"
                            + f"expl_ppo_off_s"
                            + f"/run{i}.csv"
                        )
                        ###
                        explorer = make_explorer(
                            explorer_name, alphabet, encoder, model, start_seq, rounds,
                            expmt_queries_per_round, model_queries_per_round, log_file,
                            **kwargs
                        )
                        explorer.run(landscape, verbose=True)
                        del explorer, model
                        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
