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
import fairclass.red as red
import fairclass.moments as moments
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import itertools
import run_exp
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

print = functools.partial(print, flush=True)
import xgboost as xgb
DATA_SPLIT_SEED = 4
_SMALL = True # small scale dataset for speed and testing


def lambda_response(x, a, y, learner, lamb):
    """
    Given a specified Lagrangian multiplier, find the best response
    from  logistic regression learner
    """
    n = len(a)
    p1 = len(a[a==1]) / len(a)
    p0 = len(a[a==0]) / len(a)

    # TODO: Watch out for division by zero
    # weighted protected group membership
    vec1 = lamb * a / p1
    vec2 = lamb * (1 - a) / p0
    adjust = vec1 - vec2
    cost1 =  (1 - y) + adjust  # the cost of predicting 1
    cost0 =  y
    Y = 1 * (cost0 > cost1)
    W = abs(cost0 - cost1)
    learner.fit(x, Y, W)
    pickled_learner = pickle.dumps(learner)
    f = lambda X : learner.predict(X)
    return f, pickle.loads(pickled_learner)


class LRLearner:
    """
    Basic Logistic regression baed oracle
    Oralce=LR; Class=linear
    """
    def __init__(self, C=10):
        self.regr = LogisticRegression(random_state=0, C=C,
                                       max_iter=1200,
                                       fit_intercept=False,
                                       solver='lbfgs')
        self.name = "LR Learner"

    def fit(self, X, Y, W):
        self.regr.fit(X, Y, sample_weight=W)

    def predict(self, X):
        return self.regr.predict_proba(X)


class XGBLearner:
    """
    Extreme gradient boosting classifier
    """
    def __init__(self, max_depth=3, n_estimators=150,
                 gamma=2):
        self.clf = xgb.XGBClassifier(max_depth=max_depth,
                                     silent=1,
                                     objective='binary:logistic',
                                     n_estimators=n_estimators,
                                     gamma=gamma)
        self.name = "Tree Learner"

    def fit(self, X, Y, W):
        self.clf.fit(X, Y, sample_weight=W)

    def predict(self, x):
        pred = self.clf.predict_proba(x)
        return pred




def grid_train_test(lambda_list, learner):
    """
    Take the adult dataset and get logistic regression learner from
    grid search method.

    """
    x,a,y = parser.clean_adult_full()

    if _SMALL:
        x, a, y = run_exp.subsample(x, a, y, 2000)


    x_train, a_train, y_train, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

    models = {}
    train_evaluation = {}
    test_evaluation = {}
    learners = {}
    
    for lamb in lambda_list:
        models[lamb], learners[lamb] = lambda_response(x_train, a_train, y_train, learner, lamb)
        
        train_evaluation[lamb] = evaluate.eval_BenchmarkModel(x_train,
                                                              a_train, y_train, models[lamb], "logistic")
        print(lamb, train_evaluation[lamb]['average_loss'])

        test_evaluation[lamb] = evaluate.eval_BenchmarkModel(x_test,
                                                             a_test, y_test, models[lamb], "logistic")

    result = {}
    result['learner'] = learner.name
    result['loss'] = "logistic"
    result['constraint'] = 'DP'
    result['train_eval'] = train_evaluation
    result['test_eval']  = test_evaluation
    result['learners'] = learners
    return result

# learner = LRLearner(C=10)
# grid_result = grid_train_test(np.linspace(-0.5, 0.5, 5), learner)
# # saving result
# outfile = open('adult_short_FC_lin.pkl', 'wb')
# pickle.dump(grid_result, outfile)
# outfile.close()

