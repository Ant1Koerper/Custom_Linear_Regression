import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit(self, x, y):
        # Convert x and y to NumPy arrays if they aren't already
        x = np.array(x)
        y = np.array(y)

        # Reshape x if it's a 1D array (single feature)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # Add intercept term if required
        if self.fit_intercept:
            x = np.column_stack((np.ones(x.shape[0]), x))

        # Calculate Coefficients using the Normal Equation
        x_transpose = x.T
        beta = np.linalg.pinv(x_transpose @ x) @ x_transpose @ y

        if self.fit_intercept:
            self.intercept = float(beta[0])
            self.coefficient = beta[1:]
        else:
            self.coefficient = beta

    def predict(self, x):
        # Convert x to a NumPy array
        x = np.array(x)

        # Reshape x if it's a 1D array
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if self.fit_intercept:
            x = np.column_stack((np.ones(x.shape[0]), x))
            beta = np.concatenate(([self.intercept], self.coefficient))
        else:
            beta = self.coefficient
        return x @ beta

    def r2_score(self, y, yhat):
        y = np.array(y)
        yhat = np.array(yhat)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - (ss_res / ss_tot))
        return r2

    def rmse(self, y, yhat):
        y = np.array(y)
        yhat = np.array(yhat)
        mse = np.mean((y - yhat) ** 2)
        rmse = float(np.sqrt(mse))
        return rmse
