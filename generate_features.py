import pandas as pd
import numpy as np

produce_charts = False
if produce_charts:
    import matplotlib.pyplot as plt    

def return_features(df):
    """
    Args:
        df: pandas.DataFrame, columns include at least ["date", "open", "high", "low", "close", "volume"]
    Returns:
        pandas.DataFrame
    """
    df["return"] = df["close"] / df["close"].shift(1)
    df["close_to_open"] = df["close"] / df["open"]
    df["close_to_high"] = df["close"] / df["high"]
    df["close_to_low"] = df["close"] / df["low"]
    df = df.iloc[1:] # first first row: does not have a return value
    return df

def target_value(df):
    df["y"] = df["return"].shift(-1)
    df = df.iloc[:len(df)-1]
    return df
