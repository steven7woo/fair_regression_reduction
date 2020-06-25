"""
Run the exponentiated gradient method for training a fair regression
model.

Input:
- (x, a, y): training set
- eps: target training tolerance
- Theta: the set of Threshold
- learner: the regression/classification oracle 
- constraint: for now only handles demographic parity (statistical parity)
- loss: the loss function

Output:
- a predictive model (a distribution over hypotheses)
- auxiliary model info

"""

import data_augment as augment
import fairclass.moments as moments
import fairclass.red as red


def train_FairRegression(x, a, y, eps, Theta, learner,
                                constraint="DP", loss="square", init_cache=[]):
    """
    Run fair algorithm on the training set and then record
    the metrics on the training set.

    x, a, y: the training set input for the fair algorithm
    eps: the desired level of fairness violation
    Theta: the set of thresholds (z's in the paper)
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
    else:  # exception
        raise Exception('Constraint not supported: ', str(constraint))


    print('epsilon value: ', eps, ': number of oracle calls', result.n_oracle_calls)

    model_info = {}  # dictionary for saving data
    model_info['loss_function'] = loss
    model_info['constraint'] = constraint
    model_info['exp_grad_result'] = result
    return model_info
