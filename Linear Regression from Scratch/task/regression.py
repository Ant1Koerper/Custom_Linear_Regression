import numpy as np

from custom_linear_regression import CustomLinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Compare the custom model with the linear model from scikit-learn
regSci = LinearRegression(fit_intercept=True)
reg = CustomLinearRegression(fit_intercept=True)

f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]

# Derive X
X = np.transpose(np.array([f1, f2, f3]))

# Fit the models
regSci.fit(X, y)
reg.fit(X, y)

# Predict
y_predict_sci = regSci.predict(X)
y_predict_custom = reg.predict(X)

results = {
    'Intercept': float(regSci.intercept_ - reg.intercept),
    'Coefficient': regSci.coef_ - reg.coefficient,
    'R2': r2_score(y, y_predict_sci) - reg.r2_score(y, y_predict_custom),
    'RMSE': float(mean_squared_error(y, y_predict_sci) ** 0.5 - reg.rmse(y, y_predict_custom))
}
print(results)
