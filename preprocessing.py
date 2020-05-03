import warnings
# Ignore warning: lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88 return f(*args, **kwds)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
import joblib
import generate_features

def preprocessing():
    filename = "MSFT.csv"
    df = yahoo_finance_source_to_df(filename)
    df = generate_features.return_features(df)
    df = generate_features.target_value(df)
    df = clean_df(df)
    #df = new_features(df)
    X, y = remove_unused_features(df)
    X = X.values
    y = y.values
    train_test_split = 1750
    X_learn = X[:train_test_split]
    y_learn = y[:train_test_split]
    X_test = X[train_test_split:]
    #print(X_test.shape)
    y_test = y[train_test_split:]
    assert len(X_learn) + len(X_test) == len(X)
    #print("No error raise\n")
    X_learn, scaler = scaling(X_learn)
    X_test, scaler = scaling(X_test, scaler)
    df_to_csv(df=pd.DataFrame(X_learn), filename="MSFT_X_learn.csv")
    df_to_csv(df=pd.DataFrame(y_learn), filename="MSFT_y_learn.csv")
    df_to_csv(df=pd.DataFrame(X_test), filename="MSFT_X_test.csv")
    df_to_csv(df=pd.DataFrame(y_test), filename="MSFT_y_test.csv")
    #print("Training and testing files generated\n")

def yahoo_finance_source_to_df(filename):
    """
    Lead Yahoo Finance csv and format to DataFrame
    Args:
        filename: string
    Returns:
        pandas.DataFrame
    """
    df = pd.read_csv(filename, header=0)
    df.drop(labels="Close", axis=1, inplace=True)
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    return df

def clean_df(df):
    """
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df = missing_values(df)
    df = outliers(df)
    return df

def missing_values(df):
    """
    Replace missing values with latest available
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    missing_values_count = df.isnull().sum()
    # print("Original data # missing values: {}".format(missing_values_count))
    if sum(missing_values_count) == 0:
        #print("No missing data values found\n")
        return df
    else:
        print("Ffill of missing values necessary")
        df = df.fillna(method = "ffill", axis=0).fillna("0")
        missing_values_count = df.isnull().sum()
        # print("After filling na # missing values: {}".format(missing_values_count))
        assert sum(missing_values_count) == 0
        return df

def outliers(df):
    """
    Analyze outliers of dataset
    Args:
        df: pandas.DataFrame
    Returns:
        pandas.DataFrame
    """
    df_outliers = df.loc[:,["date", "return", "close_to_open", "close_to_high", "close_to_low"]]
    column_to_analysts = "return"
    df_smallest = df_outliers.sort_values(by=column_to_analysts, ascending=True)
    df_largest = df_outliers.sort_values(by=column_to_analysts, ascending=False)
    print("Printing smallest 5 data")
    print(df_smallest.iloc[:5])
    print("\nPrinting largest 5 dataset")
    print(df_largest.iloc[:5])
    return df

def remove_unused_features(df):
    X = df.loc[1:len(df)-1, ["return","close_to_open","close_to_high","close_to_low"]]
    y = df.loc[1:len(df)-1, ["y"]]
    return X, y

def scaling(df, scaler=None):
    """
    Scale all features to unit variance and 0 mean to optimize training
    Args:
        df: pandas.DataFrame
    Returns:
        numpy.ndarray
    """
    if not scaler:
        scaler_model = sklearn_preprocessing.StandardScaler().fit(df)
        joblib.dump(scaler_model, "scaler_ml.pkl")
    else:
        scaler_model = scaler
    X = scaler_model.transform(df)
    if not scaler:
        assert sum(X.mean(axis=0)) > -0.00001 and sum(X.mean(axis=0)) < 0.00001 # zero mean
        assert sum(X.std(axis=0)) > 0.99999 and sum(X.mean(axis=0)) < 1.00001 # unit variance
    return X, scaler_model


def df_to_csv(df, filename):
    """
    Save DataFrame to csv for later use
    Args:
        df: pandas.DataFrame
        filename: string
    Returns:
        None
    """
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    preprocessing()