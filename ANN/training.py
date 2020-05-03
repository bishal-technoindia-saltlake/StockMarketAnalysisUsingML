import warnings
# Ignore warning: lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88 return f(*args, **kwds)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

def training():
    filename = "C:\\Users\\ASUS\\Desktop\\StockMarket\\MSFT.csv"
    X, y = training_data_from_csv(filename_features="C:\\Users\\ASUS\\Desktop\\StockMarket\\MSFT_X_learn.csv", 
        filename_variables="C:\\Users\\ASUS\\Desktop\\StockMarket\\MSFT_y_learn.csv")
    X = X.values # convert to numpy.ndarray used by sklearn
    y = y.values # convert to numpy.ndarray used by sklearn
    tscv = TimeSeriesSplit(n_splits=10)
    train_neural_net(X, y, tscv, False)
 

def training_data_from_csv(filename_features, filename_variables):
    """
    Load cvs file of features and target variables from csv
    Args:
        filename_features: string
        filename_variables: string
    Returns:
        pandas.DataFrame
    """
    X = pd.read_csv(filename_features, header=0)
    y = pd.read_csv(filename_variables, header=0)
    return X, y

def train_neural_net(X, y, tscv, grid_search=False):
    import neural_sklearn as nn_scikit
    nn_scikit.main(X, y, tscv.split(X))

def performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

if __name__ == "__main__":
    training()