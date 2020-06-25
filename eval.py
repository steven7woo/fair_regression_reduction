"""
Evaluate the fair model on a dataset;
Also evaluate benchmark algorithms: OLS, SEO, Logistic regression

Main function: evaluate_FairModel
Input:
- (x, a, y): evaluation set (can be training/test set)
- loss: loss function name
- result: returned by exp_grad
- Theta: the set of Threshold

Output:
- predictions over the data set
- weighted loss
- distribution over the predictions
- DP Disparity

TODO: decide the support when we compute disparity
"""

from __future__ import print_function

import functools
import numpy as np
import pandas as pd
import data_parser as parser
import data_augment as augment
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import log_loss
from scipy.stats import norm
print = functools.partial(print, flush=True)

_LOGISTIC_C = 5  # Constant for rescaled logisitic loss
_QEO_EVAL = False  # For now not handling the QEO disparity


def evaluate_FairModel(x, a, y, loss, result, Theta):
    """
    Evaluate the performance of the fair model on a dataset

    Input:
    - X, Y: augmented data
    - loss: loss function name
    - result returned by exp_grad
    - Theta: list of thresholds
    - y: original labels
    """

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment.augment_data_sq(x, a, y, Theta)
    elif loss == "absolute":  # absolute loss reweighting (uniform)
        X, A, Y, W = augment.augment_data_ab(x, a, y, Theta)
    elif loss == "logistic":  # logisitic reweighting
        X, A, Y, W = augment.augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))

    hs = result.hs
    weights = result.weights

    # first make sure the lengths of hs and weights are the same;
    off_set = len(hs) - len(weights)
    if (off_set > 0):
        off_set_list = pd.Series(np.zeros(off_set), index=[i +
                                                           len(weights)
                                                           for i in
                                                           range(off_set)])
        result_weights = weights.append(off_set_list)
    else:
        result_weights = weights

    # second filter out hypotheses with zero weights
    hs = hs[result_weights > 0]
    result_weights = result_weights[result_weights > 0]

    num_h = len(hs)
    num_t = len(Theta)
    n = int(len(X) / num_t)

    # predictions
    pred_list = [pd.Series(extract_pred(X, h(X), Theta),
                           index=range(n)) for h in hs]
    total_pred = pd.concat(pred_list, axis=1, keys=range(num_h))
    # predictions across different groups
    pred_group = extract_group_pred(total_pred, a)


    weighted_loss_vec = loss_vec(total_pred, y, result_weights, loss)

    # Fit a normal distribution to the sq_loss vector
    loss_mean, loss_std = norm.fit(weighted_loss_vec)

    # DP disp
    PMF_all = weighted_pmf(total_pred, result_weights, Theta)
    PMF_group = [weighted_pmf(pred_group[g], result_weights, Theta) for g in pred_group]
    DP_disp = max([pmf2disp(PMF_g, PMF_all) for PMF_g in PMF_group])


    # TODO: make sure at least one for each subgroup
    evaluation = {}
    evaluation['pred'] = total_pred
    evaluation['classifier_weights'] = result_weights
    evaluation['weighted_loss'] = loss_mean
    evaluation['loss_std'] = loss_std / np.sqrt(n)
    evaluation['disp_std'] = KS_confbdd(n, alpha=0.05)
    evaluation['DP_disp'] = DP_disp
    evaluation['n_oracle_calls'] = result.n_oracle_calls

    return evaluation


def eval_BenchmarkModel(x, a, y, model, loss):
    """
    Given a dataset (x, a, y) along with predictions,
    loss function name
    evaluate the following:
    - average loss on the dataset
    - DP disp
    """
    pred = model(x)  # apply model to get predictions
    n = len(y)
    if loss == "square":
        err = mean_squared_error(y, pred)  # mean square loss
    elif loss == "absolute":
        err = mean_absolute_error(y, pred)  # mean absolute loss
    elif loss == "logistic":  # assuming probabilistic predictions
        # take the probability of the positive class
        pred = pd.DataFrame(pred).iloc[:, 1]
        err = log_loss(y, pred, eps=1e-15, normalize=True)
    else:
        raise Exception('Loss not supported: ', str(loss))

    disp = pred2_disp(pred, a, y, loss)

    loss_vec = loss_vec2(pred, y, loss)
    loss_mean, loss_std = norm.fit(loss_vec)

    evaluation = {}
    evaluation['pred'] = pred
    evaluation['average_loss'] = err
    evaluation['DP_disp'] = disp['DP']
    evaluation['disp_std'] = KS_confbdd(n, alpha=0.05)
    evaluation['loss_std'] = loss_std / np.sqrt(n)

    return evaluation


def loss_vec(tp, y, result_weights, loss='square'):
    """
    Given a list of predictions and a set of weights, compute
    (weighted average) loss for each point
    """
    num_h = len(result_weights)
    if loss == 'square':
        loss_list = [(tp.iloc[:, i] - y)**2 for i in range(num_h)]
    elif loss == 'absolute':
        loss_list = [abs(tp.iloc[:, i] - y) for i in range(num_h)]
    elif loss == 'logistic':
        logistic_prob_list = [1/(1 + np.exp(- _LOGISTIC_C * (2 * tp[i]
                                                             - 1))) for i in range(num_h)]
        # logistic_prob_list = [tp[i] for i in range(num_h)]
        loss_list = [log_loss_vec(y, prob_pred, eps=1e-15) for
                     prob_pred in logistic_prob_list]
    else:
        raise Exception('Loss not supported: ', str(loss))
    df = pd.concat(loss_list, axis=1)
    weighted_loss_vec = pd.DataFrame(np.dot(df,
                                            pd.DataFrame(result_weights)))
    return weighted_loss_vec.iloc[:, 0]

def loss_vec2(pred, y, loss='square'):
    """
    Given a list of predictions and a set of weights, compute
    (weighted average) loss for each point
    """
    if loss == 'square':
        loss_vec = (pred - y)**2
    elif loss == 'absolute':
        loss_vec = abs(pred - y)
    elif loss == 'logistic':
        loss_vec = log_loss_vec(y, pred)
    else:
        raise Exception('Loss not supported: ', str(loss))
    return loss_vec


def extract_pred(X, pred_aug, Theta):
    """
    Given a list of pred over the augmented dataset, produce
    the real-valued predictions over the original dataset
    """
    width = Theta[1] - Theta[0]
    Theta_mid = Theta + (width / 2)

    num_t = len(Theta)
    n = int(len(X) / num_t)  # TODO: check whether things divide
    pred_list = [pred_aug[((j) * n):((j+1) * n)] for j in range(num_t)]
    total_pred_list = []
    for i in range(n):
        theta_index = max(0, (sum([p_vec.iloc[i] for p_vec in pred_list]) - 1))
        total_pred_list.append(Theta_mid[theta_index])
    return total_pred_list


def extract_group_pred(total_pred, a):
    """
    total_pred: predictions over the data
    a: protected group attributes
    extract the relevant predictions for each protected group
    """
    groups = list(pd.Series.unique(a))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[a == g]
    return pred_per_group


def extract_group_quantile_pred(total_pred, a, y, loss):
    """
    total_pred: a list of prediction Series
    a: protected group attributes
    y: the true label, which also gives us the quantile assignment
    """
    if loss == "logistic":
        y_quant = y  # for binary prediction task, just use labels
    else:
        y_quant = augment.quantization(y)

    groups = list(pd.Series.unique(a))
    quants = list(pd.Series.unique(y_quant))

    pred_group_quantile = {}
    pred_quantile = {}
    for q in quants:
        pred_quantile[q] = total_pred[y_quant == q]
        for g in groups:
            pred_group_quantile[(g, q)] = total_pred[(a == g) & (y_quant == q)]
    return pred_quantile, pred_group_quantile


def weighted_pmf(pred, classifier_weights, Theta):
    """
    Given a list of predictions and a set of weights, compute pmf.
    pl: a list of prediction vectors
    result_weights: a vector of weights over the classifiers
    """
    width = Theta[1] - Theta[0]
    theta_indices = pd.Series(Theta + width/2)
    weights = list(classifier_weights)
    weighted_histograms = [(get_histogram(pred.iloc[:, i],
                                          theta_indices)) * weights[i]
                           for i in range(pred.shape[1])]

    theta_counts = sum(weighted_histograms)
    pmf = theta_counts / sum(theta_counts)
    return pmf


def get_histogram(pred, theta_indices):
    """
    Given a list of discrete predictions and Theta, compute a histogram
    pred: discrete prediction Series vector
    Theta: the discrete range of predictions as a Series vector
    """
    theta_counts = pd.Series(np.zeros(len(theta_indices)))
    for theta in theta_indices:
        theta_counts[theta_indices == theta] = len(pred[pred == theta])
    return theta_counts


def pmf2disp(pmf1, pmf2):
    """
    Take two empirical PMF vectors with the same support and calculate
    the K-S stats
    """
    cdf_1 = pmf1.cumsum()
    cdf_2 = pmf2.cumsum()
    diff = cdf_1 - cdf_2
    diff = abs(diff)
    return max(diff)


def pred2_disp(pred, a, y, loss):
    """
    Input:
    pred: real-valued predictions given by the benchmark method
    a: protected group memberships
    y: labels
    loss: loss function names (for quantization)

    Output: the DP disparity of the predictions

    TODO: use the union of the predictions as the mesh
    """
    Theta = sorted(set(pred))  # find the support among the predictions
    theta_indices = pd.Series(Theta)

    if loss == "logistic":
        y_quant = y  # for binary prediction task, just use labels
    else:
        y_quant = augment.quantization(y)

    groups = list(pd.Series.unique(a))
    quants = list(pd.Series.unique(y_quant))

    # DP disparity
    histogram_all = get_histogram(pred, theta_indices)
    PMF_all = histogram_all / sum(histogram_all)
    DP_disp_group = {}
    for g in groups:
        histogram_g = get_histogram(pred[a == g], theta_indices)
        PMF_g = histogram_g / sum(histogram_g)
        DP_disp_group[g] = pmf2disp(PMF_all, PMF_g)


    disp = {}
    disp['DP'] = max(DP_disp_group.values())

    return disp


def log_loss_vec(y_true, y_pred, eps=1e-15):
    """
    return the vector of log loss over the examples
    """
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    trans_label = pd.concat([1-y_true, y_true], axis=1)
    loss = -(trans_label * np.log(y_pred)).sum(axis=1)
    return loss


def KS_confbdd(n, alpha=0.05):
    """
    Given sample size calculate the confidence interval width on K-S stats
    n: sample size
    alpha: failure prob
    ref: http://www.math.utah.edu/~davar/ps-pdf-files/Kolmogorov-Smirnov.pdf
    """
    return np.sqrt((1/(2 * n)) * np.log(2/alpha))
