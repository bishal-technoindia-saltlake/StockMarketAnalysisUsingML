import sys
sys.path.append('../..')
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def testing():
    X, y = testing_data_from_csv(filename_features="../MSFT_X_P_test.csv",
                                 filename_variables="../MSFT_Y_P_test.csv")
    X = X.values
    y = y.values
    test_neural_net(X, y)


def testing_data_from_csv(filename_features, filename_variables):
    X = pd.read_csv(filename_features, header=0)
    y = pd.read_csv(filename_variables, header=0)
    return X, y


def test_neural_net(X_test, y_test):
    X_test = PolynomialFeatures(degree=2).fit(X_test).transform(X_test)
    file_name = 'finalized_model.sav'
    model = pickle.load(open(file_name, 'rb'))
    # print(X_test.shape)
    y_pred = model.predict(X_test)
    length = np.size(y_pred)
    mse_test = mean_squared_error(y_pred, y_test)
    print("Results obtained after testing")
    print(mse_test)


if __name__ == "__main__":
    testing()
