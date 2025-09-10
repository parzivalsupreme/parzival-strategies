import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {"mae": mae, "mse": mse, "rmse": rmse}


