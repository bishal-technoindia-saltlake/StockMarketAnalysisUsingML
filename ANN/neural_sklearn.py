import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from training import performance

def main(X: np.ndarray, y: np.ndarray, tscv_idx) -> None:
    """
    Train linear Decision Tree Regressor model and save the plot of results to images/
    Args:
        X: numpy.ndarray
        y: numpy.ndarray
        tscv_idx: generator of indexes used for training and testing in folds
    Returns:
        None
    """
    X = PolynomialFeatures(degree=2).fit(X).transform(X)
    model = MLPRegressor(hidden_layer_sizes=(200, 200), solver="adam", 
        activation="relu", random_state=2)
    results = pd.DataFrame(columns=["final_train_idx", "MSE train", "MSE test"])
    i=1
    print("All the results are shown using Mean Squared Error\n")
    print("Here it is 10-fold cross validation\n")
    for train_idx, test_idx in tscv_idx:
        X_train = X[train_idx,:]
        y_train = y[train_idx,:]
        X_test = X[test_idx,:]
        y_test = y[test_idx,:]
        model.fit(X_train, y_train.ravel())
        # params = model.get_params()
        # coef = model.coef_
        y_pred = model.predict(X_train)
        print("Split",i)
        i=i+1
        print("Training result in this split")
        mse_train = performance(y_train, y_pred)
        print(mse_train)
        y_pred = model.predict(X_test)
        mse_test = performance(y_test, y_pred)
        print("Testing result in this split")
        print(mse_test)
        print("\n")
        results.loc[len(results)] = [train_idx[-1], mse_train, mse_test]
    #print(X_train.shape)
    #print(X_test.shape)
    filename='finalized_model.sav'
    pickle.dump(model,open(filename,'wb'))
    #plot_performance(results, "Neural Net scikit-learn", "neural_net_scikit")
    