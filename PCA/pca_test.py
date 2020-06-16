import sys
sys.path.append('..')
from preprocessing import *
import pandas as pd
import generate_features
from pca import PCA


def load_data():
    filename = "../MSFT.csv"
    print('loading files...')
    df = yahoo_finance_source_to_df(filename)
    df = generate_features.return_features(df)
    df = generate_features.target_value(df)
    df = clean_df(df)
    X, Y = remove_unused_features(df)
    X = X.values
    Y = Y.values
    return X, Y


def split_data(X, Y):
    print('spliting data...')
    train_test_split = 1750
    X_learn = X[:train_test_split]
    Y_learn = Y[:train_test_split]
    X_test = X[train_test_split:]
    Y_test = Y[train_test_split:]

    assert len(X_learn) + len(X_test) == len(X)

    X_learn, scaler = scaling(X_learn)
    X_test, scaler = scaling(X_test, scaler)

    df_to_csv(df=pd.DataFrame(X_learn), filename="MSFT_X_P_learn.csv")
    df_to_csv(df=pd.DataFrame(Y_learn), filename="MSFT_Y_P_learn.csv")
    df_to_csv(df=pd.DataFrame(X_test), filename="MSFT_X_P_test.csv")
    df_to_csv(df=pd.DataFrame(Y_test), filename="MSFT_Y_P_test.csv")


if __name__ == "__main__":
    X, Y = load_data()
    print('Shape of X [Before Dimension Reduction]: ', X.shape)
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)
    print('Shape of X_Projected [After Dimension Reduction] ', X_projected.shape)
    split_data(X_projected, Y)
