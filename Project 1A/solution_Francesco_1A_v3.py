# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Self-imported libraries
from sklearn.metrics import mean_squared_error
import math

def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """
    w = np.zeros((13,))

    # TODO: Enter your code here
    def ridge_obj(w):
        return ((y - X @ w).T @ (y - X @ w)) + ((w.T @ w) * lam)

    def ridge_grad(w):
        return ((X.T @ (X @ w - y)) * 2) + (w * 2 * lam)

    learning_rate = 0.00000001
    beta = 0.9
    n_steps = 100000
    tol = 1e-6

    w_curr = w - learning_rate * ridge_grad(w)
    for step in range(n_steps):
        grad = ridge_grad(w_curr)
        w_next = (1 + beta) * w_curr - beta * w - learning_rate * grad
        if np.abs(ridge_obj(w_curr) - ridge_obj(w)) < tol:
            w = w_next
            break
        w = w_curr
        w_curr = w_next

    assert w.shape == (13,)
    return w


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """

    RMSE = 0

    # TODO: Enter your code here
    MSE = 0
    y_pred = X @ w
    MSE = mean_squared_error(y,y_pred)
    RMSE = math.sqrt(MSE)

    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """
    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    # TODO: Enter your code here. Hint: Use functions 'fit' and 'calculate_RMSE' with training and test data
    # and fill all entries in the matrix 'RMSE_mat'
    col = 0
    kf = KFold(n_folds)
    for lam in lambdas:
        row = 0
        for (train, test) in enumerate(kf.split(X)):
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            w = fit(X_train,y_train,lam)
            RMSE_mat[row,col] = calculate_RMSE(w,X_test,y_test)
            row = row + 1
        col = col + 1

    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results_Mahiaandfrance_1A_v3.csv", avg_RMSE, fmt="%.12f")
