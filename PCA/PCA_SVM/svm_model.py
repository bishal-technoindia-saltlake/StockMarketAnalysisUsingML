# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle

def main(X: np.ndarray, y: np.ndarray) -> None:
    
    model= SVR(kernel='poly',degree=3,C=101,coef0=1,epsilon=0.1,tol=1e-3,gamma='auto')
    model.fit(X,y.ravel(),sample_weight=None)
    y_pred=model.predict(X)
    print("Mean Squared Error Result of training")
    print(mean_squared_error(y.ravel(),y_pred))
    filename='svm_model.save'
    pickle.dump(model,open(filename,'wb'))
    