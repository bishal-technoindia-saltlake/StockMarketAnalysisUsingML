import sys
sys.path.append('..')
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import pandas as pd
from accuracy_calculator import main as accuracy


def testing():
    X, y = testing_data_from_csv(filename_features="../MSFT_X_test.csv",
                                 filename_variables="../MSFT_y_test.csv")
    X = X.values
    y = y.values
    test_svm(X, y)


def testing_data_from_csv(filename_features, filename_variables):
    X = pd.read_csv(filename_features, header=0)
    y = pd.read_csv(filename_variables, header=0)
    return X, y


def test_svm(X_test, y_test):
    file_name = 'svm_model.save'
    model = pickle.load(open(file_name, 'rb'))
    y_pred = model.predict(X_test)
    length = np.size(y_pred)
    mse_test = mean_squared_error(y_pred, y_test.ravel())
    print("Mean Squared Error Result of Testing SVM")
    print(mse_test)
    accuracy_score = accuracy(y_pred, y_test.ravel(), length)
    print('Accuracy Score: ', accuracy_score)


if __name__ == "__main__":
    testing()
