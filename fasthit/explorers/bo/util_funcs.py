import numpy as np
from scipy.stats import norm

def UCB(preds, uncert, h_param=0.05, **kwargs):
    return preds + h_param * uncert

def LCB(preds, uncert, h_param=0.1, **kwargs):
    return preds - h_param * uncert

def EI(preds, uncert, h_param=0.1, **kwargs):
    improves = preds - kwargs["best_val"] - h_param
    z = improves / uncert
    return improves * norm.cdf(z) + uncert * norm.pdf(z)

def PI(preds, uncert, h_param=0.1, **kwargs):
    return norm.cdf(
        (preds - kwargs["best_val"] - h_param)
        / uncert
    )

def TS(preds, uncert, h_param=0.0, **kwargs):
    return kwargs["rng"].normal(preds, uncert)

def Greedy(preds, uncert, h_param=0.0, **kwargs):
    return preds