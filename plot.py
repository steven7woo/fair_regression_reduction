"""
This file contains function for visualizing experiment
results.
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
import run_exp as run_exp
from load_logged_exp import *




_PARETO = True
DATA_SPLIT_SEED = 4
uni_fontsize = 12

def disp_curve_list(result_list, base_list):
    """
    Read in a result list and trace out the disp_curve_list
    for different methods
    """
    # initialize
    constraint = "DP"
    err_bar = False
    dataset = ''

    for result in result_list:
        train_eval = result['train_eval']
        test_eval = result['test_eval']
        loss = result['loss']
        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        test_disp_dic = {}
        train_err_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}
        for eps in eps_vals:
            if constraint == "DP":
                train_disp = train_eval[eps]["DP_disp"]
                test_disp = test_eval[eps]["DP_disp"]
            elif constraint == "QEO":
                train_disp = train_eval[eps]["QEO_disp"]
                test_disp = test_eval[eps]["QEO_disp"]
            else:
                raise Exception('Constraint not supported: ', str(constraint))
            train_disp_dic[eps] = train_disp
            test_disp_dic[eps] = test_disp
            test_loss_std_dic[eps] = test_eval[eps]['loss_std']
            test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = np.sqrt(test_eval[eps]['weighted_loss'])

            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = (test_eval[eps]['weighted_loss'])


        if _PARETO:
            pareto_epsilons_train = convex_env_train(train_disp_dic,
                                                 train_err_dic)
            pareto_epsilons_test = convex_env_test(pareto_epsilons_train,
                                               test_disp_dic,
                                               test_err_dic)
        else:
            pareto_epsilons_train = eps_vals
            pareto_epsilons_test = eps_vals

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
        test_disp_list = [test_disp_dic[k] for k in pareto_epsilons_test]
        train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]
        test_err_list = [test_err_dic[k] for k in pareto_epsilons_test]

        # Getting error bars
        if loss == "square":
            err_upperconf = [np.sqrt(test_eval[k]['weighted_loss'] + 2
                                     * test_loss_std_dic[k]) - test_err_dic[k]
                             for k in pareto_epsilons_test]
            err_lowerconf = [test_err_dic[k] -
                             np.sqrt(test_eval[k]['weighted_loss'] - 2 *
                                     test_loss_std_dic[k]) for k in pareto_epsilons_test]
        else:
            err_upperconf = [2 * test_loss_std_dic[k] for k in
                             pareto_epsilons_test]
            err_lowerconf = [2 * test_loss_std_dic[k] for k in
                             pareto_epsilons_test]
        disp_conf = [test_disp_dev_dic[k] for k in
                     pareto_epsilons_test]

        if learner == 'SVM_LP':
            color = 'orange'
            err_bar = True
        elif learner == 'OLS':
            color = 'red'
            err_bar = True
        elif learner[:2] == "RF":
            color = 'brown'
            err_bar = True
        elif learner[:3] == "XGB":
            color = 'green'
            err_bar = True
        elif learner == "LR":
            color = "blue"
            err_bar = True
        else:
            color = 'tan'
            err_bar = True

        # Plotting fair model curves
        plt.subplot(1, 2, 1)
        plt.plot(train_disp_list, train_err_list, # marker="s",
                 color=color, linewidth =2, label=learner+' Fair Train')

        # plt.xlabel(constraint+' Disparity')

        if loss == "square":
            plt.ylabel('RMSE')
        elif loss == "logistic":
            plt.ylabel('Log loss')
        plt.title(dataset + ' / train')

        plt.subplot(1, 2, 2)
        if err_bar:
            plt.fill_between(np.array(test_disp_list),
                             np.array(test_err_list) -
                             np.array(err_lowerconf),
                             np.array(test_err_list) +
                             np.array(err_upperconf), alpha=0.2,
                             facecolor=color,
                             antialiased=True)

            plt.errorbar(test_disp_list, test_err_list,
                         # xerr=disp_conf,  yerr=[err_lowerconf,
                         #       err_upperconf],
                         linestyle='-', color=color,
                         label=learner+' Fair Test',  # fmt='s',
                         ecolor=color, capthick=1, markersize=5, capsize=2, linewidth=2)
        else:
            plt.plot(test_disp_list, test_err_list, # marker="s",
                     color=color, label=learner+' Fair Test', linewidth=2)

        # plt.xlabel(constraint+' Disparity')

        if loss == "square":
            plt.ylabel('RMSE')
        elif loss == "logistic":
            plt.ylabel('Log loss')
        plt.title(dataset + ' / test')

    # Plotting benchmark 
    for base_res in base_list:
        base_train_eval = base_res['base_train_eval']
        base_test_eval = base_res['base_test_eval']
        loss = base_res['loss']
        learner = base_res['learner']

        base_test_disp_conf = base_test_eval['disp_std']
        base_test_loss_std = base_test_eval['loss_std']
        dataset = base_res['dataset']

        print(learner)
        # Getting error bars
        if loss == "square":
            err_upperconf = np.sqrt(base_test_eval['average_loss'] + 2 *
                                    base_test_loss_std) - np.sqrt(base_test_eval['average_loss'])
            err_lowerconf = np.sqrt(base_test_eval['average_loss']) - np.sqrt(base_test_eval['average_loss'] - 2 *  base_test_loss_std)
            print('train loss', np.sqrt(base_train_eval['average_loss']))
            print('test loss', np.sqrt(base_test_eval['average_loss']))
            print('2 loss std', err_lowerconf, err_upperconf)
        else:
            err_upperconf = 2 * base_test_loss_std
            err_lowerconf = 2 * base_test_loss_std
            print('train loss', (base_train_eval['average_loss']))
            print('test loss', (base_test_eval['average_loss']))
            print('2 loss std', err_lowerconf, err_upperconf)


        marker = 'o'

        if learner == 'OLS':
            color = 'red'
            err_bar = True
        elif learner == "SEO":
            color = 'deepskyblue'
            err_bar = False
            marker = '^'
        elif learner[:2] == "RF":
            color = 'brown'
            err_bar = True
        elif learner[:3] == "XGB":
            color = "green"
            err_bar = True
        elif learner == "LR":
            color = "blue"
            err_bar = True
        else:
            color = 'tan'
            err_bar = False
        plt.subplot(1, 2, 1)
        if loss == "square":
            plt.scatter([base_train_eval[constraint+'_disp']],
                        [np.sqrt(base_train_eval['average_loss'])],
                        marker=marker, color=color, label=
                        'Unconstrained '+learner+' ')
            plt.ylabel('RMSE')
        else:
            plt.scatter([base_train_eval[constraint+'_disp']],
                        [(base_train_eval['average_loss'])],
                        marker=marker, color=color, label='Unconstrained '+learner+' ')
            plt.ylabel('log loss')

        plt.title(dataset + ' / train') 

        plt.subplot(1, 2, 2)
        if loss == "square":
            if err_bar:
                plt.errorbar([base_test_eval[constraint+'_disp']],
                             [np.sqrt(base_test_eval['average_loss'])],
                             xerr=base_test_disp_conf,# yerr=[[err_lowerconf], [err_upperconf]],
                             marker="o",
                             color=color, label= 'Unconstrained '+learner+' ', ecolor='black', capthick=1, markersize=6, capsize=2)
            else:
                plt.scatter([base_test_eval[constraint+'_disp']],
                            [np.sqrt(base_test_eval['average_loss'])],
                            marker=marker, color=color, label= 'Unconstrained '+learner+' ')
            plt.ylabel('RMSE')
        else:
            if err_bar:
                plt.errorbar([base_test_eval[constraint+'_disp']],
                             [base_test_eval['average_loss']],
                             xerr=base_test_disp_conf,# yerr=[[err_lowerconf], [err_upperconf]],
                             marker="o",
                             color=color, label= 'Unconstrained '+learner+' ', ecolor='black', capthick=1, markersize=6, capsize=2)
            else:
                plt.scatter([base_test_eval[constraint+'_disp']],
                            [(base_test_eval['average_loss'])],
                            marker=marker, color=color, label='Unconstrained '+learner+' ')
            plt.ylabel('log loss')
        # plt.ylim(0.20, 0.4)
        # plt.legend()
        plt.title(dataset + ' / test')
    plt.show()



def disp_test_res(result_list, base_list, full=True, paired_test=True):
    """
    Read in a result list and trace out the disp_curve_list
    for different methods
    """
    # initialize
    constraint = "DP"
    err_bar = False

    # First calcuate the baseline loss vector
    base_res = base_list[0]
    dataset = base_res['dataset']
    if dataset == 'law_school':
        x, a, y = parser.clean_lawschool_full()
    elif dataset == 'communities':
        x, a, y = parser.clean_communities_full()
    elif dataset == 'adult':
        x, a, y = parser.clean_adult_full()
    if not full:
        x, a, y = run_exp.subsample(x, a, y, 2000)

    _, _, _, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
    n = len(y_test)
    loss = base_res['loss']
    if paired_test:
        base_pred = base_res['base_test_eval']['pred']
        base_loss_vec = evaluate.loss_vec2(base_pred, y_test, loss)
        base_mean_loss = np.mean(base_loss_vec)
        print(base_mean_loss)


    for result in result_list:
        train_eval = result['train_eval']
        test_eval = result['test_eval']

        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        test_disp_dic = {}
        train_err_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}

        linestyle = '-'
        if learner == 'SVM_LP':
            color = 'orange'
            err_bar = True
        elif learner == 'OLS':
            color = 'red'
            err_bar = True
        elif learner[:2] == "RF":
            color = 'brown'
            err_bar = True
        elif learner == "XGB Classifier":
            color = 'blue'
            err_bar = True
            linestyle = '--'
        elif learner == "XGB Regression":
            color = 'red'
            err_bar = True
            linestyle = '--'
        elif learner == "LR":
            color = "blue"
            err_bar = True
        else:
            color = 'tan'
            err_bar = True

        for eps in eps_vals:
            if constraint == "DP":
                train_disp = train_eval[eps]["DP_disp"]
                test_disp = test_eval[eps]["DP_disp"]
            elif constraint == "QEO":
                train_disp = train_eval[eps]["QEO_disp"]
                test_disp = test_eval[eps]["QEO_disp"]
            else:
                raise Exception('Constraint not supported: ', str(constraint))
            train_disp_dic[eps] = train_disp
            test_disp_dic[eps] = test_disp

            if paired_test:
                test_total_pred = test_eval[eps]['pred']

                test_res_weights = test_eval[eps]['classifier_weights']
                weighted_loss_vec = evaluate.loss_vec(test_total_pred, y_test,
                                                      test_res_weights, loss)
                diff_vec = weighted_loss_vec - base_loss_vec
                loss_mean, loss_std = norm.fit(diff_vec)
                test_loss_std_dic[eps] = loss_std / np.sqrt(n)
            else:
                test_loss_std_dic[eps] = test_eval[eps]['loss_std']

            test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = np.sqrt(test_eval[eps]['weighted_loss']) - np.sqrt(base_mean_loss)
            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = (test_eval[eps]['weighted_loss'] - base_mean_loss)

        if _PARETO:
            pareto_epsilons_train = convex_env_train(train_disp_dic,
                                                     train_err_dic)
            pareto_epsilons_test = convex_env_test(pareto_epsilons_train,
                                                   test_disp_dic,
                                                   test_err_dic)
        else:
            pareto_epsilons_train = eps_vals
            pareto_epsilons_test = eps_vals

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
        test_disp_list = [test_disp_dic[k] for k in pareto_epsilons_test]
        train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]
        test_err_list = [test_err_dic[k] for k in pareto_epsilons_test]

        # Getting error bars
        if loss == "square":
            err_upperconf = [np.sqrt(test_eval[k]['weighted_loss'] + 2
                                     * test_loss_std_dic[k]) -  np.sqrt(test_eval[k]['weighted_loss']) for k in
                             pareto_epsilons_test]


            err_lowerconf = [np.sqrt(test_eval[k]['weighted_loss']) - np.sqrt(test_eval[k]['weighted_loss'] - 2 * test_loss_std_dic[k])  for k in
                             pareto_epsilons_test]
        else:
            err_upperconf = [2 * test_loss_std_dic[k] for k in
                             pareto_epsilons_test]
            err_lowerconf = [2 * test_loss_std_dic[k] for k in
                             pareto_epsilons_test]
            disp_conf = [test_disp_dev_dic[k] for k in
                     pareto_epsilons_test]


        plt.fill_between(np.array(test_disp_list),
                         np.array(test_err_list) -
                         np.array(err_lowerconf),
                         np.array(test_err_list) +
                         np.array(err_upperconf), alpha=0.2,
                         facecolor=color, antialiased=True)

        plt.errorbar(test_disp_list, test_err_list, color=color,
                     capthick=1, markersize=5, capsize=2,
                     linewidth=2, linestyle=linestyle)

    # Plotting benchmark 
    for base_res in base_list:
        base_train_eval = base_res['base_train_eval']
        base_test_eval = base_res['base_test_eval']
        loss = base_res['loss']
        learner = base_res['learner']

        base_test_disp_conf = base_test_eval['disp_std']
        base_test_loss_std = base_test_eval['loss_std']
        dataset = base_res['dataset']

        marker = '^'
        label = 'unconstrained'

        if learner == 'OLS':
            # color = 'red'
            color = 'darksalmon'
            err_bar = True
        elif learner == "SEO":
            marker = 'v'
            color = 'deepskyblue'
            err_bar = True
            label='SEO'
        elif learner[:2] == "RF":
            color = 'brown'
            err_bar = True
        elif learner[:3] == "XGB":
            color = "green"
            err_bar = True
        elif learner == "LR":
            color = "darksalmon"
            err_bar = True
        else:
            color = 'tan'
            err_bar = False

        # Getting error bars
        if loss == "square":
            err_upperconf = np.sqrt(base_test_eval['average_loss'] + 2 *
                                    base_test_loss_std) - np.sqrt(base_test_eval['average_loss'])
            err_lowerconf = np.sqrt(base_test_eval['average_loss']) - np.sqrt(base_test_eval['average_loss'] - 2 *  base_test_loss_std)
        else:
            err_upperconf = 2 * base_test_loss_std
            err_lowerconf = 2 * base_test_loss_std
    
        if loss == "square":
            if err_bar:
                plt.errorbar([base_test_eval[constraint+'_disp']],
                             [np.sqrt(base_test_eval['average_loss']) - np.sqrt(base_mean_loss)],
                             xerr=base_test_disp_conf,
                             marker=marker, markeredgecolor = 'black',
                             color='black', markerfacecolor=color, ecolor='black', capthick=1, markersize=11, capsize=2)
            else:
                plt.scatter([base_test_eval[constraint+'_disp']],
                            [np.sqrt(base_test_eval['average_loss']) - np.sqrt(base_mean_loss)],
                            marker=marker, edgecolors = 'black', s=95,
                            label= label)
        else:
            if err_bar:
                plt.errorbar([base_test_eval[constraint+'_disp']],
                             [base_test_eval['average_loss']] - base_mean_loss,
                             xerr=base_test_disp_conf,
                             marker=marker,  markeredgecolor = 'black', markerfacecolor=color,
                             color='black', ecolor='black', capthick=1, markersize=11, capsize=2)
            else:
                plt.scatter([base_test_eval[constraint+'_disp']],
                            [(base_test_eval['average_loss'])] - base_mean_loss,  
                            marker=marker, label=label)




def disp_train_res(result_list, base_list):
    """
    Read in a result list and trace out the disp_curve_list
    for different methods
    """
    # initialize
    constraint = "DP"
    err_bar = False

    for result in result_list:
        train_eval = result['train_eval']
        test_eval = result['test_eval']

        
        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        test_disp_dic = {}
        train_err_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}

        linestyle = '-'
        if learner == 'SVM_LP':
            color = 'orange'
        elif learner == 'OLS':
            color = 'red'
        elif learner[:2] == "RF":
            color = 'brown'
        elif learner == "XGB Classifier":
            color = 'blue'
            linestyle = '--'
        elif learner == "XGB Regression":
            color = 'red'
            linestyle = '--'
        elif learner == "LR":
            color = "blue"
        else:
            color = 'tan'

        for eps in eps_vals:

            train_disp = train_eval[eps]["DP_disp"]
            test_disp = test_eval[eps]["DP_disp"]
            train_disp_dic[eps] = train_disp
            test_disp_dic[eps] = test_disp

            loss = result['loss']
            
            test_loss_std_dic[eps] = test_eval[eps]['loss_std']
            test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])

            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])

        pareto_epsilons_train = convex_env_train(train_disp_dic,
                                                     train_err_dic)

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
        train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]

        plt.plot(train_disp_list, train_err_list, color=color, linewidth=2, linestyle=linestyle)

    # Plotting benchmark 
    for base_res in base_list:
        base_train_eval = base_res['base_train_eval']
        loss = base_res['loss']
        learner = base_res['learner']
        dataset = base_res['dataset']

        marker = '^'
        label = 'unconstrained'

        if learner == 'OLS':
            # color = 'red'
            color = 'darksalmon'
        elif learner == "SEO":
            marker = 'v'
            color = 'deepskyblue'
            label='SEO'
        elif learner[:2] == "RF":
            color = 'brown'
        elif learner[:3] == "XGB":
            color = "green"
        elif learner == "LR":
            color = "darksalmon"
        else:
            color = 'tan'
  
        if loss == "square":
            plt.scatter([base_train_eval[constraint+'_disp']],
                            [np.sqrt(base_train_eval['average_loss'])],
                            marker=marker, color=color,  edgecolors = 'black', s=95)
        else:
            plt.scatter([base_train_eval[constraint+'_disp']],
                        [(base_train_eval['average_loss'])],
                        marker=marker, color=color, edgecolors =
                        'black', s=95)



def disp_grid_search_train(result):
    """
    This function is specific to plotting the grid search method.
    """
    # initialize
    constraint = "DP"
    err_bar = False

    train_eval = result['train_eval']
    test_eval = result['test_eval']
    constraint = result['constraint']
    learner = result['learner']
    eps_vals = train_eval.keys()
    train_disp_dic = {}
    test_disp_dic = {}
    train_err_dic = {}
    test_err_dic = {}
    test_loss_std_dic = {}
    test_disp_dev_dic = {}

    if learner[:2] == "LR":
        linestyle='-'
    else:
        linestyle = '--'

    for eps in eps_vals:
        train_disp = train_eval[eps]["DP_disp"]
        test_disp = test_eval[eps]["DP_disp"]
        train_disp_dic[eps] = train_disp
        test_disp_dic[eps] = test_disp

        loss = result['loss']
            
        test_loss_std_dic[eps] = test_eval[eps]['loss_std']
        test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

        train_err_dic[eps] = (train_eval[eps]['average_loss'])

    pareto_epsilons_train = convex_env_train(train_disp_dic,
                                             train_err_dic)




    # taking the pareto frontier
    train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
    train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]

    plt.plot(train_disp_list, train_err_list, color='black', linewidth=2, linestyle=linestyle)



def disp_grid_search_test(result, base_list, full=True, paired_test=True):
    """
    This function is specific to plotting grid search method
    """
    # initialize
    constraint = "DP"
    err_bar = False

    # First calcuate the baseline loss vector
    base_res = base_list[0]
        
    x, a, y = parser.clean_adult_full()
    if not full:
        x, a, y = run_exp.subsample(x, a, y, 2000)

    _, _, _, x_test, a_test, y_test = run_exp.train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
    n = len(y_test)
    loss = base_res['loss']
    if paired_test:
        base_pred = base_res['base_test_eval']['pred']
        base_loss_vec = evaluate.loss_vec2(base_pred, y_test, loss)
        base_mean_loss = np.mean(base_loss_vec)

    train_eval = result['train_eval']
    test_eval = result['test_eval']

    constraint = result['constraint']
    learner = result['learner']
    eps_vals = train_eval.keys()
    train_disp_dic = {}
    test_disp_dic = {}
    train_err_dic = {}
    test_err_dic = {}
    test_loss_std_dic = {}
    test_disp_dev_dic = {}

    if learner[:2] == "LR":
        linestyle='-'
    else:
        linestyle = '--'

    for eps in eps_vals:
        if constraint == "DP":
            train_disp = train_eval[eps]["DP_disp"]
            test_disp = test_eval[eps]["DP_disp"]
        else:
            raise Exception('Constraint not supported: ', str(constraint))
        train_disp_dic[eps] = train_disp
        test_disp_dic[eps] = test_disp

        if paired_test:
            test_total_pred = test_eval[eps]['pred']
            loss_vec = evaluate.loss_vec2(test_total_pred, y_test, loss)
            diff_vec = loss_vec - base_loss_vec
            loss_mean, loss_std = norm.fit(diff_vec)
            test_loss_std_dic[eps] = loss_std / np.sqrt(n)
        else:
            test_loss_std_dic[eps] = test_eval[eps]['loss_std']

        test_disp_dev_dic[eps] = test_eval[eps]['disp_std']


        train_err_dic[eps] = (train_eval[eps]['average_loss'])
        test_err_dic[eps] = (test_eval[eps]['average_loss'] - base_mean_loss)

    if _PARETO:
        pareto_epsilons_train = convex_env_train(train_disp_dic,
                                                 train_err_dic)
        pareto_epsilons_test = convex_env_test(pareto_epsilons_train,
                                               test_disp_dic,
                                               test_err_dic)
    else:
        pareto_epsilons_train = eps_vals
        pareto_epsilons_test = eps_vals

    # taking the pareto frontier
    train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
    test_disp_list = [test_disp_dic[k] for k in pareto_epsilons_test]
    train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]
    test_err_list = [test_err_dic[k] for k in pareto_epsilons_test]

    err_upperconf = [2 * test_loss_std_dic[k] for k in
                     pareto_epsilons_test]
    err_lowerconf = [2 * test_loss_std_dic[k] for k in
                     pareto_epsilons_test]
    disp_conf = [test_disp_dev_dic[k] for k in pareto_epsilons_test]

    print(pareto_epsilons_test)

    plt.fill_between(np.array(test_disp_list),
                         np.array(test_err_list) -
                         np.array(err_lowerconf),
                         np.array(test_err_list) +
                         np.array(err_upperconf), alpha=0.2,
                         facecolor='black', antialiased=True)

    plt.errorbar(test_disp_list, test_err_list, color='black',
                     capthick=1, markersize=5, capsize=2,
                     linewidth=2, linestyle=linestyle)


def find_pareto_frontier_dict(Xs, Ys):
    """
    Find the pareto curve using the dict data structures Xs and Ys: are
    dictionaries that share the same set of keys return a subset of
    keys

    """
    # Sort the list in either ascending or descending order of the
    # items values in Xs
    key_X_pairs = sorted(Xs.items(), key=lambda x: x[1],
                         reverse=False)  # this is a list of (key, val) pairs
    # Start the Pareto frontier with the first key value in the sorted list
    p_front = [key_X_pairs[0][0]]
    # Loop through the sorted list
    for (key, X) in key_X_pairs:
        if Ys[key] <= Ys[p_front[-1]]:  # Look for lower values of Y
            p_front.append(key)
    return p_front


def convex_env_train(Xs, Ys):
    """
    Identify the convex envelope on the set of models
    from the train set.
    """
    # Sort the list in either ascending or descending order of the
    # items values in Xs
    key_X_pairs = sorted(Xs.items(), key=lambda x: x[1],
                         reverse=False)  # this is a list of (key, val) pairs
    # Start the Pareto frontier with the first key value in the sorted list
    p_front = [key_X_pairs[0][0]]
    # Loop through the sorted list
    count = 0
    for (key, X) in key_X_pairs:
        if Ys[key] <= Ys[p_front[-1]]:  # Look for lower values of Y
            if count > 0:
                p_front.append(key)
        count = count + 1
    return remove_interior(p_front, Xs, Ys)
    

def convex_env_test(pareto_epsilons_train, test_disp_dic, test_err_dic):
    Xs = {k : test_disp_dic[k] for k in pareto_epsilons_train}
    Ys = {k : test_err_dic[k] for k in pareto_epsilons_train}
    return convex_env_train(Xs, Ys)

def remove_interior(p_front, Xs, Ys):
    if len(p_front) < 3:
        return p_front
    [k1, k2, k3] = p_front[:3]
    x1 = Xs[k1]
    y1 = Ys[k1]
    x2 = Xs[k2]
    y2 = Ys[k2]
    x3 = Xs[k3]
    y3 = Ys[k3]
    # compute the linear interpolation between 1 and 3 when x = x2
    if x1 == x3:  # equal values 
        return remove_interior([k1] + p_front[3:], Xs, Ys)
    else:
        alpha = (x2 - x1) / (x3 - x1)
        y_hat = y1 - (y1 - y3) * alpha
        if y_hat >= y2:  # keep the triplet
            return [k1] + remove_interior(p_front[1:], Xs, Ys)
        else:  # remove 2
            return remove_interior([k1, k3]+p_front[3:], Xs, Ys)

def plot_Ncalls(result_list):
    # initialize
    constraint = "DP"
    err_bar = False

    # print(result_list[0]['learner'], base_list[0]['learner'])

    for result in result_list:
        train_eval = result['train_eval']
        test_eval = result['test_eval']

        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        test_disp_dic = {}
        train_err_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}

        linestyle = '-'
        if learner == 'SVM_LP':
            color = 'orange'
            label = 'Fair reg. (oracle=CS, class=linear)'
        elif learner == 'OLS':
            color = 'red'
            label = 'Fair reg. (oracle=LS, class=linear)'
        elif learner[:2] == "RF":
            color = 'brown'
        elif learner == "XGB Classifier":
            color = 'blue'
            linestyle = '--'
        elif learner == "XGB Regression":
            color = 'red'
            linestyle = '--'
        elif learner == "LR":
            color = "blue"
            label = 'Fair reg. (oracle=LR, class=linear)'
        else:
            color = 'tan'

        n_calls = {}
        for eps in eps_vals:

            train_disp = train_eval[eps]["DP_disp"]
            test_disp = test_eval[eps]["DP_disp"]
            n_calls[eps] = train_eval[eps]['n_oracle_calls']

            train_disp_dic[eps] = train_disp
            test_disp_dic[eps] = test_disp
            test_loss_std_dic[eps] = test_eval[eps]['loss_std']
            test_disp_dev_dic[eps] = test_eval[eps]['disp_std']
            train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
            test_err_dic[eps] = (test_eval[eps]['weighted_loss'])

        pareto_epsilons_train = eps_vals
        pareto_epsilons_test = eps_vals

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in pareto_epsilons_train]
        test_disp_list = [test_disp_dic[k] for k in pareto_epsilons_test]
        train_err_list = [train_err_dic[k] for k in pareto_epsilons_train]
        test_err_list = [test_err_dic[k] for k in pareto_epsilons_test]

        plt.plot(eps_vals, list(n_calls.values()), color = color, linestyle=linestyle, marker='o', label=label)
        print(n_calls)

    plt.legend()

def plot_orcale_calls():
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(20, 18))
    fig.text(0.5, 0.01, 'specified disparity', ha='center', fontsize=uni_fontsize)
    plt.subplot(131)
    plt.ylabel('# oracle calls', fontsize=uni_fontsize)
    plot_Ncalls([lawschool_Ncalls_OLS, lawschool_Ncalls_SVM])
    plt.title('law school subsampled', fontsize=uni_fontsize)

    plt.subplot(132)
    plot_Ncalls([adult_Ncalls_OLS, adult_Ncalls_SVM, adult_Ncalls_Logistic])
    plt.title('adult subsampled', fontsize=uni_fontsize)

    plt.subplot(133)
    plot_Ncalls([comm_Ncalls_OLS, comm_Ncalls_SVM])
    plt.title('communities & crime', fontsize=uni_fontsize)
    plt.show()


def plot_singles():
    fig, ax = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True, figsize=(20, 18))
    fig.text(0.5, 0.08, 'DP disparity', ha='center', fontsize=12)

    plt.subplot(531)
    plt.ylim(0.29, 0.45)
    disp_test_res([adult_OLS], [adult_bl[0]])
    plt.ylabel("Log Loss", fontsize=12)
    plt.title('adult / least square')

    plt.subplot(532)
    plt.ylim(0.29, 0.45)
    disp_test_res([adult_Logistic], [adult_bl[0]])
    plt.title('adult / log. reg.')

    plt.subplot(533)
    plt.ylim(0.29, 0.45)
    disp_test_res([adult_XGB], [adult_bl[1]])
    plt.title('adult / grad. boosting')

    plt.subplot(534)
    plt.ylim(0.092, 0.116)
    disp_test_res([lawschool_OLS], [lawschool_bl[0], lawschool_bl[3]])
    plt.ylabel("RMSE", horizontalalignment='left' , fontsize=12)
    plt.title('law school / least square')

    plt.subplot(535)
    plt.ylim(0.092, 0.116)
    disp_test_res([lawschool_RF], [lawschool_bl[1]])
    plt.title('law school / random forest')

    plt.subplot(536)
    plt.ylim(0.092, 0.116)
    disp_test_res([lawschool_XGB], [lawschool_bl[2]])
    plt.title('law school / grad. boosting')

    plt.subplot(537)
    plt.ylim(0.13, 0.22)
    disp_test_res([comm_OLS], [comm_bl[0], comm_bl[2]])
    plt.ylabel("relative RMSE", horizontalalignment='left', fontsize=12)
    plt.title('communities & crime / least square')

    plt.subplot(538)
    plt.ylim(0.13, 0.22)
    disp_test_res([comm_SVM], [comm_bl[0], comm_bl[2]])
    plt.title('communities & crime / SVM')

    plt.subplot(539)
    plt.ylim(0.13, 0.22)
    disp_test_res([comm_RF], [comm_bl[1]])
    plt.title('communities & crime / random forest')

    plt.subplot(5, 3, 10)
    plt.ylim(0.35, 0.52)
    disp_test_res([adult_short_OLS], [adult_short_bl[0]], full=False)
    plt.ylabel("Log Loss", horizontalalignment='left', fontsize=12)
    plt.title('adult small / least square')

    plt.subplot(5, 3, 11)
    plt.ylim(0.35, 0.52)
    disp_test_res([adult_short_SVM], [adult_short_bl[0]], full=False)
    plt.title('adult small / SVM')

    plt.subplot(5, 3, 12)
    plt.ylim(0.35, 0.52)
    disp_test_res([adult_short_Logistic], [adult_short_bl[0]], full=False)
    plt.title('adult small / log. reg.')

    plt.subplot(5, 3, 13)
    plt.ylim(0.098, 0.112)
    disp_test_res([lawschool_short_OLS], [lawschool_short_bl[0], lawschool_short_bl[1]], full=False)
    plt.ylabel("RMSE", horizontalalignment='left', fontsize=12)
    plt.title('law school small / least square')

    plt.subplot(5, 3, 14)
    plt.ylim(0.098, 0.112)
    disp_test_res([lawschool_short_SVM], [lawschool_short_bl[0], lawschool_short_bl[1]], full=False)
    plt.legend(loc=10, bbox_to_anchor=(1.6, 0.4), fontsize=12)
    plt.title('law school small / SVM')

    plt.savefig('test.pdf', bbox_inches='tight')
    plt.show()



def plot_multiples_test():
    fig, _ = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 18))
    # fig.text(0.5, 0.03, 'SP disparity', ha='center', fontsize=uni_fontsize)
    # fig.text(0.51, 0.06, 'SP disparity', ha='center', fontsize=uni_fontsize*0.9)
    # fig.text(0.8, 0.06, 'SP disparity', ha='center', fontsize=uni_fontsize*0.9)


    plt.subplot(231)
    plt.ylim(-0.005, 0.07)
    disp_test_res([comm_OLS, comm_SVM], [comm_bl[0], comm_bl[2]])
    plt.title('communities & crime ')
    plt.ylabel('relative RMSE', fontsize=uni_fontsize)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)


    plt.subplot(232)
    plt.ylim(-0.001, 0.01)
    disp_test_res([lawschool_short_OLS, lawschool_short_SVM], [lawschool_short_bl[0], lawschool_short_bl[1]], full=False)
    plt.title('law school subsampled', fontsize=uni_fontsize)

    plt.subplot(233)
    plt.ylim(-0.002, 0.025)
    disp_test_res([lawschool_OLS, lawschool_XGB], [lawschool_bl[2], lawschool_bl[3], lawschool_bl[0]])
    plt.title('law school', fontsize=uni_fontsize)


    plt.subplot(2, 3, 5)
    plt.ylim(-0.01, 0.11)
    disp_test_res([adult_short_OLS, adult_short_SVM, adult_short_Logistic], [adult_short_bl[0]], full=False)

    disp_grid_search_test(adult_short_FC_lin, [adult_short_bl[0]], full=False, paired_test=True)
    plt.title('adult subsampled', fontsize=uni_fontsize)
    plt.ylabel('relative log loss', fontsize=uni_fontsize)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)

    ax = plt.subplot(2, 3, 6)
    plt.ylim(-0.01, 0.15)
    disp_test_res([adult_OLS, adult_Logistic , adult_XGB, adult_LS_XGB], [adult_bl[1], adult_bl[0]])
    disp_grid_search_test(adult_FC_lin, [adult_bl[1], adult_bl[0]], full=True, paired_test=True)
    disp_grid_search_test(adult_FC_tree, [adult_bl[1], adult_bl[0]], full=True, paired_test=True)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)
    plt.title('adult', fontsize=uni_fontsize)



    ax.plot([0.1], [100], color='orange',linewidth=2,  markersize=5, label='fair reg. (oracle=CS, model=linear)')
    ax.plot([0.1], [100], color='red', linewidth=2, markersize=5, label='fair reg. (oracle=LS, model=linear)')
    ax.plot([0.1], [100], color='red', linewidth=2, linestyle='--', markersize=5, label='fair reg. (oracle=LS, model=tree ensemble)')


    ax.errorbar([0.1], [100], xerr=[0.0], marker='^', markeredgecolor
                 = 'black', color='black', markerfacecolor='darksalmon',
                 label='unconstrained reg. (model=linear)', ecolor='black',
                 capthick=1, markersize=11, capsize=2)
    ax.errorbar([0.1], [100], xerr=[0.0], marker='^', markeredgecolor
                 = 'black', color='black', markerfacecolor='green',
                 label='unconstrained reg. (model=tree ensemble)', ecolor='black',
                 capthick=1, markersize=11, capsize=2)

    ax.scatter([0.1], [100], marker='v', color='deepskyblue', edgecolors = 'black', s=95,
                label='SEO (model=linear)')

    ax.plot([0.1], [100], color='blue', linewidth=2, markersize=5, label='fair reg. (oracle=LR, model=linear)')
    ax.plot([0.1], [100], color='blue', linestyle='--', linewidth=2, markersize=5, label='fair reg. (oracle=LR, model=tree ensemble)')
    ax.plot([0.1], [100], color='black', linestyle='--', linewidth=2, markersize=5, label='fair class. (oracle=LR, model=tree ensemble)')
    ax.plot([0.1], [100], color='black', linestyle='-', linewidth=2, markersize=5, label='fair class. (oracle=LR, model=linear)')

    ax.plot([0.1], [100], color='white', label=' ')
    ax.plot([0.1], [100], color='white', label='Bottom plots only:')

    handles, labels = ax.get_legend_handles_labels()
    handles2 = [handles[0], handles[1], handles[2], handles[9], handles[10], handles[11], handles[7], handles[8], handles[3], handles[4], handles[6], handles[5]]
    labels2 = [labels[0], labels[1], labels[2], labels[9], labels[10], labels[11], labels[7], labels[8], labels[3], labels[4], labels[6], labels[5]]
    ax.legend(handles2, labels2, loc=10, bbox_to_anchor=(-2.1, 0.5), fontsize=10)
    plt.show()



def plot_multiples_train():
    fig, _= plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 18))
    # fig.text(0.5, 0.03, 'SP disparity', ha='center', fontsize=uni_fontsize)
    # fig.text(0.51, 0.06, 'SP disparity', ha='center', fontsize=uni_fontsize*0.9)
    # fig.text(0.8, 0.06, 'SP disparity', ha='center', fontsize=uni_fontsize*0.9)
    # fig.text(0.23, 0.48, 'SP disparity', ha='center', fontsize=uni_fontsize*0.9)

    plt.subplot(231)
    disp_train_res([comm_OLS, comm_SVM], [comm_bl[0], comm_bl[2]])
    plt.title('communities & crime ')
    plt.ylabel('RMSE', fontsize=uni_fontsize)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)

    plt.subplot(232)
    disp_train_res([lawschool_short_OLS, lawschool_short_SVM], [lawschool_short_bl[0], lawschool_short_bl[1]])
    plt.title('law school subsampled', fontsize=uni_fontsize)
    
    plt.subplot(233)
    disp_train_res([lawschool_OLS, lawschool_XGB], [lawschool_bl[2], lawschool_bl[3], lawschool_bl[0]])
    plt.title('law school', fontsize=uni_fontsize)

    plt.subplot(2, 3, 5)

    disp_train_res([adult_short_OLS, adult_short_SVM, adult_short_Logistic], [adult_short_bl[0]])

    disp_grid_search_train(adult_short_FC_lin)

    plt.title('adult subsampled', fontsize=uni_fontsize)
    plt.ylabel('log loss', fontsize=uni_fontsize)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)

    ax=plt.subplot(2, 3, 6)
    plt.ylim(0.25, 0.45)
    disp_train_res([adult_OLS, adult_Logistic, adult_XGB, adult_LS_XGB], [adult_bl[1], adult_bl[0]])
    disp_grid_search_train(adult_FC_lin)
    disp_grid_search_train(adult_FC_tree)  
    plt.title('adult ', fontsize=uni_fontsize)
    plt.xlabel('SP disparity', fontsize=uni_fontsize*0.9)

    ax.plot([0.1], [100], color='orange',linewidth=2,  markersize=5, label='fair reg. (oracle=CS, model=linear)')
    ax.plot([0.1], [100], color='red', linewidth=2, markersize=5, label='fair reg. (oracle=LS, model=linear)')
    ax.plot([0.1], [100], color='red', linewidth=2, linestyle='--', markersize=5, label='fair reg. (oracle=LS, model=tree ensemble)')

    ax.errorbar([0.1], [100], xerr=[0.0], marker='^', markeredgecolor
                 = 'black', color='black', markerfacecolor='darksalmon',
                 label='unconstrained reg. (model=linear)', ecolor='black',
                 capthick=1, markersize=11, capsize=2)
    ax.errorbar([0.1], [100], xerr=[0.0], marker='^', markeredgecolor
                 = 'black', color='black', markerfacecolor='green',
                 label='unconstrained reg. (model=tree ensemble)', ecolor='black',
                 capthick=1, markersize=11, capsize=2)
    ax.scatter([0.1], [100], marker='v', color='deepskyblue', edgecolors = 'black', s=95,
                label='SEO (model=linear)')

    ax.plot([0.1], [100], color='blue', linewidth=2, markersize=5, label='fair reg. (oracle=LR, model=linear)')
    ax.plot([0.1], [100], color='blue', linestyle='--', linewidth=2, markersize=5, label='fair reg. (oracle=LR, model==tree ensemble)')

    ax.plot([0.1], [100], color='black', linestyle='--', linewidth=2, markersize=5, label='fair class. (oracle=LR, model=tree ensemble)')
    ax.plot([0.1], [100], color='black', linestyle='-', linewidth=2, markersize=5, label='fair class. (oracle=LR, model=linear)')

    ax.plot([0.1], [100], color='white', label=' ')
    ax.plot([0.1], [100], color='white', label='Bottom plots only:')

    handles, labels = ax.get_legend_handles_labels()
    handles2 = [handles[0], handles[1], handles[2], handles[9], handles[10], handles[11], handles[7], handles[8], handles[3], handles[4], handles[6], handles[5]]
    labels2 = [labels[0], labels[1], labels[2], labels[9], labels[10], labels[11], labels[7], labels[8], labels[3], labels[4], labels[6], labels[5]]
    ax.legend(handles2, labels2, loc=10, bbox_to_anchor=(-2.1, 0.5), fontsize=10)

    plt.show()


plot_multiples_test()
plot_multiples_train()


