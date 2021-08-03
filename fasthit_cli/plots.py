import glob
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from typing import Optional, Tuple

def cumulative_max_per_round(sequences):
    num_rounds = sequences["round"].max() + 1
    max_per_round = [sequences["true_score"][sequences["round"] == r].max()
                     for r in range(num_rounds)]
    return np.maximum.accumulate(max_per_round)

def compute_scores(run_dir):
    n_measurements = []
    measurements = []
    scores = []
    for fname in glob.glob(f'{run_dir}/run*.csv'):
        # Skip metadata in header
        with open(fname) as f:
            next(f)
            df = pd.read_csv(f)

        max_per_round = cumulative_max_per_round(df)
        scores.extend(max_per_round)
        measurements.extend(range(len(max_per_round)))
        n_measurements.extend(sorted(set(df["measurement_cost"].tolist())))
    
    return pd.DataFrame(
        {
            "Round": measurements,
            "Number of samples": n_measurements,
            "Cumulative maximum": scores,
        }
    )

def sns_plot(
    names_and_dirs: str,
    save_name: Optional[str] = None,
    x_is_round: bool = True,
):
    fig, ax = plt.subplots(dpi=300)
    sns.set_style('whitegrid')
    n_measurements_max = 0
    for name, run_dir in names_and_dirs.items():
        data = compute_scores(run_dir)
        rounds = data["Round"].to_numpy()
        n_measurements = data["Number of samples"].to_numpy()
        cum_maxes = data["Cumulative maximum"].to_numpy()
        max_round = rounds.max() + 1
        n_measurements_max = max(n_measurements.max(), n_measurements_max)
        n_repeat = (len(rounds) - 1) // max_round + 1
        metric = []
        for i in range(n_repeat):
            start = i*max_round
            end = min(len(rounds), (i+1)*max_round)
            round = rounds[start:end]
            cum_max = cum_maxes[start:end]
            metric.append(metrics.auc(round, cum_max))
        metric = np.mean(np.array(metric))
        if x_is_round:
            sns.lineplot(
                x="Round", y="Cumulative maximum", data=data,
                label="(AUC:{:.2f})".format(metric) + name, linewidth=1
            )
        else:
            sns.lineplot(
                x="Number of samples", y="Cumulative maximum", data=data,
                label=name, linewidth=1
            )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if x_is_round:
        plt.xlim(0, max_round - 1)
    else:
        plt.xlim(0, n_measurements_max)
    plt.ylim(0.0, 1.01)
    plt.legend()
    ###
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

def line_plot(
    names_and_dirs: str,
    save_name: Optional[str] = None,
):
    for name, run_dir in names_and_dirs.items():
        fig, ax = plt.subplots(dpi=300)
        data = compute_scores(run_dir)
        rounds = data["Round"].to_numpy()
        cum_maxes = data["Cumulative maximum"].to_numpy()
        max_round = rounds.max() + 1
        n_repeat = (len(rounds) - 1) // max_round + 1
        max_loc = Counter()
        for i in range(n_repeat):
            start = i*max_round
            end = min(len(rounds), (i+1)*max_round)
            round = rounds[start:end]
            cum_max = cum_maxes[start:end]
            max_loc.update([np.argmax(cum_max)])
            ax.plot(round, cum_max, linewidth=1)#, label=f"seed:{i+1}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlim(0, max_round - 1)
        plt.ylim(0.0, 1.01)
        plt.xlabel("Round")
        plt.ylabel("Cumulative maximum")
        #plt.legend()
        for loc, cnt in max_loc.items():
            plt.text(loc, 1.0, str(cnt))
        ###
        if save_name is not None:
            path, fmt = save_name.split(".")
            plt.savefig(f"{path}_{name}.{fmt}")
        plt.show()

def plot(
    names_and_dirs: str,
    save_name: Optional[str] = None,
):
    fig, ax = plt.subplots(dpi=300)
    sns.set_style('whitegrid')
    n_measurements_max = 0
    for name, run_dir in names_and_dirs.items():
        data = compute_scores(run_dir)
        rounds = data["Round"].to_numpy()
        n_measurements = data["Number of samples"].to_numpy()
        cum_maxes = data["Cumulative maximum"].to_numpy()
        max_round = rounds.max() + 1
        n_measurements_max = max(n_measurements.max(), n_measurements_max)
        n_repeat = (len(rounds) - 1) // max_round + 1
        for i in range(n_repeat):
            start = i*max_round
            end = min(len(rounds), (i+1)*max_round)
            round = rounds[start:end]
            cum_max = cum_maxes[start:end]
        ax.plot(round, cum_max, linewidth=1, label=name)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(0, max_round - 1)
    plt.ylim(0.0, 1.01)
    plt.legend()
    ###
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
        