from scipy.stats import ncx2
import numpy as np


def compute_inductive_bias(model_errors, target_error):
    df, nc, loc, scale = ncx2.fit(model_errors, floc=0)
    return inductive_bias(target_error, df, nc, loc, scale)


def inductive_bias(error, df, nc, loc, scale):
    # Scaled Chi-Squared
    log_prob = -ncx2.logcdf(error, df, nc=nc, loc=loc, scale=scale)
    if np.isinf(log_prob):
        log_prob = inductive_bias_alt(error, df, nc, loc, scale)
    return log_prob


def inductive_bias_alt(error, df, nc, loc, scale):
    # Normalize by scale
    error = error / scale

    h = 1 - 2 / 3 * (df + nc) * (df + 3 * nc) / (df + 2 * nc) ** 2
    p = (df + 2 * nc) / (df + nc) ** 2
    m = (h - 1) * (1 - 3 * h)
    z_score = ((error / (df + nc)) ** h
               - (1 + h * p * (h - 1 - 0.5 * (2 - h) * m * p))) / (h * np.sqrt(2 * p) * (1 + 0.5 * m * p))
    return z_score ** 2 / 2
