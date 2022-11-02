import pandas as pd
import numpy as np
import editdistance as edd


def calc_hd(ref_seq, variants):
    return [edd.eval(ref_seq, variant) for variant in variants]

def starting_points(landscape_file, wt_seq, intervals):
    data = pd.read_csv(landscape_file)
    top_seq = data.loc[data["Fitness"].argmax(), "Variants"]
    wt_fitness = data[data["Variants"] == wt_seq]["Fitness"]
    
    hd = calc_hd(top_seq, data["Variants"])
    data["HD"] = hd
    data_hd = [data.loc[data["HD"] == i] for i in range(1, 5)]

    data_fitness = []
    for data in data_hd:
        data1 = data.loc[(data["Fitness"] >= intervals[0]) & (data["Fitness"] < intervals[1])]
        data_fitness.append(data1.reset_index(drop=True))
        data2 = data.loc[(data["Fitness"] >= intervals[1]) & (data["Fitness"] < intervals[2])]
        data_fitness.append(data2.reset_index(drop=True))
        data3 = data.loc[(data["Fitness"] >= intervals[2]) & (data["Fitness"] < intervals[3])]
        data_fitness.append(data3.reset_index(drop=True))
        data4 = data.loc[(data["Fitness"] >= intervals[3]) & (data["Fitness"] < intervals[4])]
        data_fitness.append(data4.reset_index(drop=True))
        data5 = data.loc[(data["Fitness"] >= intervals[4]) & (data["Fitness"] < intervals[5])]
        data_fitness.append(data5.reset_index(drop=True))
    
    np.random.seed(1)
    starts = pd.DataFrame([data.loc[np.random.randint(0, len(data))] for data in data_fitness if not data.empty])
    starts.reset_index(drop=True)
    print(starts)