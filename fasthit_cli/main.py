import numpy as np

import fasthit
from fasthit.utils import utils
import fasthit.utils.sequence_utils as s_utils


def main():
    seed = 42
    rounds = 10
    expmt_queries_per_round = 384
    model_queries_per_round = 3200
    kwargs = {
        "ac_func": "LCB",
        "kernel": "RBF",
    }
    ###
    np.random.seed(seed)
    utils.set_torch_seed(seed)
    ###
    problem = fasthit.landscapes.gb1.registry()["with_imputed"]
    landscape = fasthit.landscapes.GB1(**problem["params"])
    alphabet = s_utils.AAS
    ###
    encoder = fasthit.encoders.OneHot(alphabet)
    ###
    model = fasthit.models.GPRegressor(kernel=kwargs["kernel"])
    log_file = (
        f"runs_gb1_test/with_imputed/"
        + f"bo_evo/one-hot/gpr/"
        + f"restart_0"
        + f"/run0.csv"
    )
    ###
    start_seq = problem["starts"][0]
    explorer = fasthit.explorers.BO_EVO(
        encoder,
        model,
        rounds=rounds,
        expmt_queries_per_round=expmt_queries_per_round,
        model_queries_per_round=model_queries_per_round,
        starting_sequence=start_seq,
        alphabet=alphabet,
        log_file=log_file,
        proposal_func=kwargs["ac_func"],
    )
    ###
    explorer.run(landscape, verbose=False)

if __name__ == "__main__":
    main()