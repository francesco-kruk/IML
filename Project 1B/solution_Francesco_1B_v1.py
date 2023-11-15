# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))

    # TODO: Enter your code here
    X_transformed[:, 0] = X[:, 0]
    X_transformed[:, 1] = X[:, 1]
    X_transformed[:, 2] = X[:, 2]
    X_transformed[:, 3] = X[:, 3]
    X_transformed[:, 4] = X[:, 4]
    X_transformed[:, 5] = np.square(X[:, 0])
    X_transformed[:, 6] = np.square(X[:, 1])
    X_transformed[:, 7] = np.square(X[:, 2])
    X_transformed[:, 8] = np.square(X[:, 3])
    X_transformed[:, 9] = np.square(X[:, 4])
    X_transformed[:, 10] = np.exp(X[:, 0])
    X_transformed[:, 11] = np.exp(X[:, 1])
    X_transformed[:, 12] = np.exp(X[:, 2])
    X_transformed[:, 13] = np.exp(X[:, 3])
    X_transformed[:, 14] = np.exp(X[:, 4])
    X_transformed[:, 15] = np.cos(X[:, 0])
    X_transformed[:, 16] = np.cos(X[:, 1])
    X_transformed[:, 17] = np.cos(X[:, 2])
    X_transformed[:, 18] = np.cos(X[:, 3])
    X_transformed[:, 19] = np.cos(X[:, 4])
    X_transformed[:, 20] = 1

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)

    # TODO: Enter your code here
    """
    def obj(w):
        return ((y - X_transformed @ w).T @ (y - X_transformed @ w) * 0.5)

    def obj_grad(w):
        return (X_transformed.T @ (X_transformed @ w - y))

    learning_rate = 0.0005
    beta = 0.9
    n_steps = 1000000
    tol = 1e-6

    w_curr = w - learning_rate * obj_grad(w)
    for step in range(n_steps):
        grad = obj_grad(w_curr)
        w_next = (1 + beta) * w_curr - beta * w - learning_rate * grad
        if np.abs(obj(w_curr) - obj(w)) < tol:
            print("Found it!")
            w = w_next
            break
        w = w_curr
        w_curr = w_next
    """
    learning_rate = 0.00000001
    beta = 0.9
    n_steps = 100000
    tol = 1e-6
    lam = 200

    def ridge_obj(w):
        return ((y - X_transformed @ w).T @ (y - X_transformed @ w)) + ((w.T @ w) * lam)

    def ridge_grad(w):
        return ((X_transformed.T @ (X_transformed @ w - y)) * 2) + (w * 2 * lam)

    w_curr = w - learning_rate * ridge_grad(w)
    for step in range(n_steps):
        grad = ridge_grad(w_curr)
        w_next = (1 + beta) * w_curr - beta * w - learning_rate * grad
        if np.abs(ridge_obj(w_curr) - ridge_obj(w)) < tol:
            print("Found it!")
            w = w_next
            break
        w = w_curr
        w_curr = w_next

    assert w.shape == (21,)
    return w


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    # print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit(X, y)
    # Save results in the required format
    np.savetxt("./results_Francesco_1B_v1.csv", w, fmt="%.12f")
