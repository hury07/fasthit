import numpy as np
from scipy.stats import norm


def UCB(preds, uncertainties, _best_fitness):
    kappa = 0.05
    return preds + kappa * uncertainties

def LCB(preds, uncertainties, _best_fitness):
    kappa = 0.05
    return preds - kappa * uncertainties

def TS(preds, uncertainties, _best_fitness):
    return np.random.normal(preds, uncertainties)

def EI(preds, uncertainties, _best_fitness):
    eps = 0.1
    improves = preds - _best_fitness - eps
    z = improves / uncertainties
    return improves * norm.cdf(z) + uncertainties * norm.pdf(z)

def PI(preds, uncertainties, _best_fitness):
    eps = 0.2
    return norm.cdf(
        (preds - _best_fitness - eps)
        / uncertainties
    )

def UCE(preds, uncertainties, _best_fitness):
    return uncertainties

def Greedy(preds, uncertainties, _best_fitness):
    return preds