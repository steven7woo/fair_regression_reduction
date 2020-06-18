"""
Run the exponentiated gradient method for training a fair regression
model.

Input:
- (x, a, y): training set
- eps: target training tolerance
- Theta: the set of Threshold

Output:
distribution over hypotheses


Also provide a collection of functions for evaluating the output model.
"""

from __future__ import print_function

import functools
import numpy as np
import pandas as pd
import data_parser as parser
import data_augment as augment
import solvers as solvers
import eval as evaluate
import fairclass.moments as moments
import fairclass.red as red

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

print = functools.partial(print, flush=True)

_LOGISTIC_C = 5  # Constant for rescaled logistic loss


def train_FairRegression(x, a, y, eps, Theta, learner,
                                constraint="DP", loss="square", init_cache=[]):
    """
    First run the fair algorithm on the training set and then record
    the metrics on thre training set.

    x, a, y: the training set input for the fair algorithm
    eps: the desired level of fairness violation

    """
    alpha = (Theta[1] - Theta[0])/2

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment.augment_data_sq(x, a, y, Theta)
    elif loss == "absolute":  # absolute loss reweighting (uniform)
        X, A, Y, W = augment.augment_data_ab(x, a, y, Theta)
    elif loss == "logistic":  # logisitic reweighting
        X, A, Y, W = augment.augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))

    if constraint == "DP":  # DP constraint
        result = red.expgrad(X, A, Y, learner, dataW=W,
                             cons_class=moments.DP_theta, eps=eps,
                             debug=False, init_cache=init_cache)
    elif constraint == "QEO":  # QEO constraint; currently not supported
        result = red.expgrad(X, A, Y, learner, dataW=W,
                             cons_class=moments.QEO, eps=eps, debug=True, init_cache=init_cache)
    else:  # exception
        raise Exception('Constraint not supported: ', str(constraint))


    print('epsilon value: ', eps, ': number of oracle calls', result.n_oracle_calls)

    model_info = {}  # dictionary for saving data
    model_info['loss_function'] = loss
    model_info['constraint'] = constraint
    model_info['exp_grad_result'] = result
    return model_info
