"""
Two types of solvers/optimizers:

1. The first type take in an augmented data set returned by
data_augment, and try to minimize classification error over the
following hypothesis class: { h(X) = 1[ f(x) >= x['theta']] : f in F}
over some real-valued class F.

Input: augmented data set, (X, Y, W)
Output: a model that can predict label Y

These solvers are used with exp_grad

2. The second type simply solves the regression problem
on a data set (x, a, y)

These solvers serve as our unconstrained benchmark methods.
"""



import functools
import numpy as np
import pandas as pd
import random
import data_parser as parser
import data_augment as augment
from gurobipy import *

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
import time


_LOGISTIC_C = 5  # Constant for rescaled logisitic loss; might have to
                 # change for data_augment
# from sklearn.model_selection import train_test_split

"""
Oracles for fair regression algorithm
"""
class SVM_LP_Learner:
    """
    Gurobi based cost-sensitive classification oracle
    Assume there is a 'theta' field in the X data frame
    Oracle=CS; Class=linear
    """
    def __init__(self, off_set=0, norm_bdd=1):
        self.weights = None
        self.norm_bdd = norm_bdd  # initialize the norm bound to be 2
        self.off_set = off_set
        self.name = 'SVM_LP'

    def fit(self, X, Y, W):
        w = SVM_Gurobi(X, Y, W, self.norm_bdd, self.off_set)
        self.weights = pd.Series(w, index=list(X.drop(['theta'], 1)))

    def predict(self, X):
        y_values = (X.drop(['theta'],
                           axis=1)).dot(np.array(self.weights))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class LeastSquaresLearner:
    """
    Basic Least regression square based oracle
    Oracle=LS; class=linear
    """
    def __init__(self, Theta):
        self.weights = None
        self.Theta = Theta
        self.name = "OLS"

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=None)
        self.weights = pd.Series(self.lsqinfo[0], index=list(matX))

    def predict(self, X):
        y_values = (X.drop(['theta'],
                           axis=1)).dot(np.array(self.weights))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

class LogisticRegressionLearner:
    """
    Basic Logistic regression baed oracle
    Oralce=LR; Class=linear
    """
    def __init__(self, Theta, C=10000, regr=None):
        self.Theta = Theta
        self.name = "LR"

        if regr is None:
            self.regr = LogisticRegression(random_state=0, C=C,
                                           max_iter=1200,
                                           fit_intercept=False,
                                           solver='lbfgs')
        else:
            self.regr = regr


    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY, sample_weight=vecW)
        pred_prob = self.regr.predict_proba(matX)

    def predict(self, X):
        pred_prob = self.regr.predict_proba(X.drop(['theta'], axis=1))
        prob_values = pd.DataFrame(pred_prob)[1]
        y_values = (np.log(1 / prob_values - 1) / (- _LOGISTIC_C) + 1) / 2
        # y_values = pd.DataFrame(pred_prob)[1]
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

class RF_Classifier_Learner:
    """
    Basic RF classifier based CSC
    Oracle=LR; Class=Tree ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "RF Classifier"
        self.clf = RandomForestClassifier(max_depth=4,
                                           random_state=0,
                                           n_estimators=20)

    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.clf.fit(matX, vecY, sample_weight=vecW)

    def predict(self, X):
        pred_prob = self.clf.predict_proba(X.drop(['theta'],
                                                   axis=1))
        y_values = pd.DataFrame(pred_prob)[1]
        pred = 1*(y_values - X['theta'] >= 0)
        return pred

class XGB_Classifier_Learner:
    """
    Basic GB classifier based oracle
    Oracle=LR; Class=Tree ensemble
    """
    def __init__(self, Theta, clf=None):
        self.Theta = Theta
        self.name = "XGB Classifier"
        param = {'max_depth' : 3, 'silent' : 1, 'objective' :
                 'binary:logistic', 'n_estimators' : 150, 'gamma' : 2}
        if clf is None:
            self.clf = xgb.XGBClassifier(**param)
        else:
            self.clf = clf

    def fit(self, X, Y, W):
        matX, vecY, vecW = approx_data_logistic(X, Y, W, self.Theta)
        self.clf.fit(matX, vecY, sample_weight=vecW)

    def predict(self, X):
        pred_prob = self.clf.predict_proba(X.drop(['theta'],
                                                  axis=1))
        prob_values = pd.DataFrame(pred_prob)[1]
        y_values = (np.log(1 / prob_values - 1) / (- _LOGISTIC_C) + 1) / 2
        pred = 1*(y_values - X['theta'] >= 0)
        return pred

class RF_Regression_Learner:
    """
    Basic random forest based oracle
    Oracle=LS; Class=Tree ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "RF Regression"
        self.regr = RandomForestRegressor(max_depth=4, random_state=0,
                                          n_estimators=200)

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class XGB_Regression_Learner:
    """
    Gradient boosting based oracle
    Oracle=LS; Class=Tree Ensemble
    """
    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "XGB Regression"
        params = {'max_depth': 4, 'silent': 1, 'objective':
                  'reg:linear', 'n_estimators': 200, 'reg_lambda' : 1,
                  'gamma':1}
        self.regr = xgb.XGBRegressor(**params)

    def fit(self, X, Y, W):
        matX, vecY = approximate_data(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

# HELPER FUNCTIONS HERE FOR BestH Oracles
def SVM_Gurobi(X, Y, W, norm_bdd, off_set):
    """
    Solving SVM using Gurobi solver
    X: design matrix with the last two columns being 'theta'
    A: protected feature
    impose ell_infty constraint over the coefficients
    """
    d = len(X.columns) - 1  # number of predictive features (excluding theta)
    N = X.shape[0]  # number of augmented examples
    m = Model()
    m.setParam('OutputFlag', 0)    
    Y_aug = Y.map({1: 1, 0: -1})
    # Add a coefficient variable per feature
    w = {}
    for j in range(d):
        w[j] = m.addVar(lb=-norm_bdd, ub=norm_bdd,
                        vtype=GRB.CONTINUOUS, name="w%d" % j)
    w = pd.Series(w)

    # Add a threshold value per augmented example
    t = {}  # threshold values
    for i in range(N):
        t[i] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="t%d" % i)
    t = pd.Series(t)
    m.update()
    for i in range(N):
        xi = np.array(X.drop(['theta'], 1).iloc[i])
        yi = Y_aug.iloc[i]
        theta_i = X['theta'][i]
        # Hinge Loss Constraint
        m.addConstr(t[i] >=  off_set - (w.dot(xi) - theta_i) * yi)
    m.setObjective(quicksum(t[i] * W.iloc[i] for i in range(N)))
    m.optimize()
    weights = np.array([w[i].X for i in range(d)])
    return np.array(weights)


def approximate_data(X, Y, W, Theta):
    """
    Given the augmented data (X, Y, W), recover for each example the
    prediction in Theta + alpha/2 that minimizes the cost;
    Thus we reduce the size back to the same orginal size
    """
    n = int(len(X) / len(Theta))  # size of the dataset
    alpha = (Theta[1] - Theta[0])/2
    x = X.iloc[:n, :].drop(['theta'], 1)
    pred_vec = Theta + alpha  # the vector of possible preds

    minimizer = {}

    pred_vec = {}  # mapping theta to pred vector
    for pred in (Theta + alpha):
        pred_vec[pred] = (1 * (pred >= pd.Series(Theta)))

    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]  # the set of rows for i-th example
        W_i = W.iloc[index_set]
        Y_i = Y.iloc[index_set]
        Y_i.index = range(len(Y_i))
        W_i.index = range(len(Y_i))
        cost_i = {}
        for pred in (Theta + alpha):
            cost_i[pred] = abs(Y_i - pred_vec[pred]).dot(W_i)
        minimizer[i] = min(cost_i, key=cost_i.get)
    return x, pd.Series(minimizer)


def approx_data_logistic(X, Y, W, Theta):
    """
    Given the augmented data (X, Y, W), recover for each example the
    prediction in Theta + alpha/2 that minimizes the cost;
    Then create a pair of weighted example so that the prob pred
    will minimize the log loss.
    """
    n = int(len(X) / len(Theta))  # size of the dataset
    alpha = (Theta[1] - Theta[0])/2
    x = X.iloc[:n, :].drop(['theta'], 1)

    pred_vec = {}  # mapping theta to pred vector
    Theta_mid = [0] + list(Theta + alpha) + [1]
    Theta_mid = list(filter(lambda x: x >= 0, Theta_mid))
    Theta_mid = list(filter(lambda x: x <= 1, Theta_mid))

    for pred in Theta_mid:
        pred_vec[pred] = (1 * (pred >= pd.Series(Theta)))

    minimizer = {}
    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]  # the set of rows for i-th example
        W_i = W.iloc[index_set]
        Y_i = Y.iloc[index_set]
        Y_i.index = range(len(Y_i))
        W_i.index = range(len(Y_i))
        cost_i = {}
        for pred in Theta_mid:  # enumerate different possible
                                      # predictions
            cost_i[pred] = abs(Y_i - pred_vec[pred]).dot(W_i)
        minimizer[i] = min(cost_i, key=cost_i.get)

    matX = pd.concat([x]*2, ignore_index=True)
    y_1 = pd.Series(1, np.arange(len(x)))
    y_0 = pd.Series(0, np.arange(len(x)))
    vecY = pd.concat([y_1, y_0], ignore_index=True)
    w_1 = pd.Series(minimizer)
    w_0 = 1 - pd.Series(minimizer)
    vecW = pd.concat([w_1, w_0], ignore_index=True)
    return matX, vecY, vecW


"""
SECOND CLASS OF BENCHMARK SOLVERS
"""

class OLS_Base_Learner:
    """
    Basic OLS solver
    """
    def __init__(self):
        self.regr = linear_model.LinearRegression(fit_intercept=False)
        self.name = "OLS"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        pred = self.regr.predict(x)
        return pred


class SEO_Learner:
    """
    SEO learner by JFS
    """
    def __init__(self):
        self.weights_SEO = None
        self.name = "SEO"

    def fit(self, x, y, sens_attr):
        """
        assume sens_attr is contained in x
        """
        lsqinfo_SEO = np.linalg.lstsq(x, y, rcond=None)
        weights_SEO = pd.Series(lsqinfo_SEO[0], index=list(x))
        self.weights_SEO = weights_SEO.drop(sens_attr)

    def predict(self, x, sens_attr):
        x_res = x.drop(sens_attr, 1)
        pred = x_res.dot(self.weights_SEO)
        return pred


class Logistic_Base_Learner:
    """
    Simple logisitic regression
    """
    def __init__(self, C=10000):
        # use liblinear smaller datasets
        self.regr = LogisticRegression(random_state=0, C=C,
                                       max_iter=1200,
                                       fit_intercept=False,
                                       solver='lbfgs')
        self.name = "LR"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # probabilistic predictions
        pred = self.regr.predict_proba(x)
        return pred

class RF_Base_Regressor: 
    """
    Standard Random Forest Regressor
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=20):
        # initialize a rf learner
        self.regr = RandomForestRegressor(max_depth=max_depth,
                                          random_state=0,
                                          n_estimators=n_estimators)
        self.name = "RF Regressor"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # predictions
        pred = self.regr.predict(x)
        return pred

class RF_Base_Classifier: 
    """
    Standard Random Forest Classifier
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=20):
        # initialize a rf learner
        self.regr = RandomForestClassifier(max_depth=max_depth,
                                           random_state=0,
                                           n_estimators=n_estimators)
        self.name = "RF Classifier"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        # predictions
        pred = self.regr.predict_proba(x)
        return pred

class XGB_Base_Classifier:
    """
    Extreme gradient boosting classifier
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=3, n_estimators=150,
                 gamma=2):
        self.clf = xgb.XGBClassifier(max_depth=max_depth,
                                     silent=1,
                                     objective='binary:logistic',
                                     n_estimators=n_estimators,
                                     gamma=gamma)
        self.name = "XGB Classifier"

    def fit(self, x, y):
        self.clf.fit(x, y)

    def predict(self, x):
        pred = self.clf.predict_proba(x)
        return pred


class XGB_Base_Regressor:
    """
    Extreme gradient boosting regressor
    This is for baseline evaluation; not for fair learn oracle
    """
    def __init__(self, max_depth=4, n_estimators=200):
        param = {'max_depth': max_depth, 'silent': 1, 'objective':
                 'reg:linear', 'n_estimators': n_estimators, 'reg_lambda' : 1, 'gamma':1}
        self.regr = xgb.XGBRegressor(**param)
        self.name = "XGB Regressor"

    def fit(self, x, y):
        self.regr.fit(x, y)

    def predict(self, x):
        pred = self.regr.predict(x)
        return pred


def runtime_test():
    """
    Testing the runtime for different oracles
    Taking 1000 examples from the law school dataset.
    """
    x, a, y = parser.clean_lawschool_full()
    x, a, y = parser.subsample(x, a, y, 1000)
    Theta = np.linspace(0, 1.0, 21)
    X, A, Y, W = augment.augment_data_sq(x, a, y, Theta)
    alpha = (Theta[1] - Theta[0])/2

    start = time.time()
    learner1 = SVM_LP_Learner(off_set=alpha, norm_bdd=1)
    learner1.fit(X, Y, W)
    end = time.time()
    print("SVM", end - start)

    start = time.time()
    learner2 = LeastSquaresLearner(Theta)
    learner2.fit(X, Y, W)
    end = time.time()
    print("OLS", end - start)

    start = time.time()
    learner3 = LogisticRegressionLearner(Theta)
    learner3.fit(X, Y, W)
    end = time.time()
    print("Logistic", end - start)

    start = time.time()
    learner4 = XGB_Regression_Learner(Theta)
    learner4.fit(X, Y, W)
    end = time.time()
    print("XGB least square", end - start)

    start = time.time()
    learner5 = XGB_Classifier_Learner(Theta)
    learner5.fit(X, Y, W)
    end = time.time()
    print("XGB logistic", end - start)
