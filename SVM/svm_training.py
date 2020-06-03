import pandas as pd
import numpy as np
import warnings
import sys
sys.path.append('..')


def training():
    X, y = training_data_from_csv(filename_features="../MSFT_X_learn.csv",
                                  filename_variables="../MSFT_y_learn.csv")
    X = X.values
    y = y.values
    train_svm_model(X, y)


def training_data_from_csv(filename_features, filename_variables):
    """
        Load csv file of features and target variables from csv
        Args:
            filename_features: string
            filename_variables: string
        Returns:
            pandas.DataFrame
    """
    X = pd.read_csv(filename_features, header=0)
    y = pd.read_csv(filename_variables, header=0)
    return X, y


def train_svm_model(X, y):
    import svm_model as svm_scikit
    svm_scikit.main(X, y)


if __name__ == "__main__":
    training()
