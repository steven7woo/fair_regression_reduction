"""
Take the regression result and threshold the scores
to obtain binary predictions
"""

from __future__ import print_function
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import eval as evaluate
import solvers as solvers
import exp_grad as fairlearn
import run_exp as run_exp
from scipy.stats import norm
import data_parser as parser
print = functools.partial(print, flush=True)
import data_parser as parser
from load_logged_exp import *

from plot import convex_env_train, convex_env_test

DATA_SPLIT_SEED = 4

def score2pred(reg_res, thresh=0.5, full=False):
    train_eval = reg_res['train_eval']
    test_eval = reg_res['test_eval']
    
    eps_vals = train_eval.keys()

    train_err = {}
    test_err = {}
    train_disp = {}
    test_disp = {}

    x, a, y = parser.clean_adult_full()
    if not full:
        x, a, y = run_exp.subsample(x, a, y, 2000)

    x_train, a_train,  y_train, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

    for eps in eps_vals:
        class_train_pred = 1 *  (train_eval[eps]['pred'] >= thresh) 
        class_test_pred = 1 * (test_eval[eps]['pred'] >= thresh) 
        train_clf_weights = train_eval[eps]['classifier_weights']
        test_clf_weights = test_eval[eps]['classifier_weights']

        train_clf_weights.index = range(len(train_clf_weights))
        test_clf_weights.index = range(len(test_clf_weights))

        train_pred = (class_train_pred.dot(train_clf_weights))
        test_pred = (class_test_pred.dot(test_clf_weights))
        avg_train_pred = np.mean(train_pred)
        avg_train_pred0 = np.mean(train_pred[a_train==0])
        avg_train_pred1 = np.mean(train_pred[a_train==1])

        avg_test_pred = np.mean(test_pred)
        avg_test_pred0 = np.mean(test_pred[a_test==0])
        avg_test_pred1 = np.mean(test_pred[a_test==1])

        train_disp[eps] = max(abs(avg_train_pred - avg_train_pred0), abs(avg_train_pred - avg_train_pred1))
        test_disp[eps] = max(abs(avg_test_pred - avg_test_pred0), abs(avg_test_pred - avg_test_pred1))
        
        train_err[eps] = sum(abs(train_pred - y_train)) / len(y_train)
        test_err[eps] = sum(abs(test_pred - y_test)) / len(y_test)

    return train_disp, train_err, test_disp, test_err





def grid_score2pred(result, thresh=0.5, full=False):
    train_eval = result['train_eval']
    test_eval = result['test_eval']
    lamb_vals = train_eval.keys()

    grid_train_disp = {}
    grid_test_disp = {}
    grid_train_err = {}
    grid_test_err = {}


    x, a, y = parser.clean_adult_full()
    if not full:
        x, a, y = run_exp.subsample(x, a, y, 2000)

    x_train, a_train,  y_train, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)

    for lamb in lamb_vals:
        train_pred = train_eval[lamb]['pred']
        train_thres_pred = 1 * (train_pred > thresh)

        test_pred = test_eval[lamb]['pred']
        test_thres_pred = 1 * (test_pred > thresh)
        
        avg_train_pred = np.mean(train_thres_pred)
        avg_train_pred0 = np.mean(train_thres_pred[a_train==0])
        avg_train_pred1 = np.mean(train_thres_pred[a_train==1])

        avg_test_pred = np.mean(test_thres_pred)
        avg_test_pred0 = np.mean(test_thres_pred[a_test==0])
        avg_test_pred1 = np.mean(test_thres_pred[a_test==1])
        
        grid_train_disp[lamb] = max(abs(avg_train_pred - avg_train_pred0), abs(avg_train_pred - avg_train_pred1))
        grid_test_disp[lamb] = max(abs(avg_test_pred - avg_test_pred0), abs(avg_test_pred - avg_test_pred1))
        
        grid_train_err[lamb] = sum(abs(train_pred - y_train)) / len(y_train)
        grid_test_err[lamb] = sum(abs(test_pred - y_test)) / len(y_test)

    return grid_train_disp, grid_train_err, grid_test_disp, grid_test_err




def plot_thresh_res(regr_res, FC_res, thresh=0.5, full=False):
    train_disp, train_err, test_disp, test_err = score2pred(regr_res, thresh, full)
    grid_train_disp, grid_train_err, grid_test_disp, grid_test_err = grid_score2pred(FC_res, thresh, full)

    train_disp_list = [train_disp[k] for k in convex_env_train(train_disp, train_err)]
    train_err_list = [train_err[k] for k in convex_env_train(train_disp, train_err)]

    grid_train_disp_list = [grid_train_disp[k] for k in convex_env_train(grid_train_disp, grid_train_err)]
    grid_train_err_list = [grid_train_err[k] for k in convex_env_train(grid_train_disp, grid_train_err)]
    plt.plot(train_disp_list, train_err_list, label="Thresholding fair regression")
    # plt.plot(test_disp.values(), test_err.values())

    plt.plot(grid_train_disp_list, grid_train_err_list, label="Thresholding fair classification")
    # plt.plot(grid_test_disp.values(), grid_test_err.values())
    plt.legend()
    plt.show()



plot_thresh_res(adult_short_Logistic, adult_short_FC_lin, thresh=0.5)
plot_thresh_res(adult_XGB, adult_FC_tree, full=True, thresh=0.5)
plot_thresh_res(adult_Logistic, adult_FC_lin, full=True, thresh=0.5)
