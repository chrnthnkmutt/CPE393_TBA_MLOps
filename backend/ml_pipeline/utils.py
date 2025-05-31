
import numpy as np
import pandas as pd

def add_custom_features(df):
    df = df.copy()
    df["has_cap_gain"] = (df["capital.gain"] > 0).astype(int)
    df["has_cap_loss"] = (df["capital.loss"] > 0).astype(int)
    df["gain_loss_ratio"] = (df["capital.gain"] + 1) / (df["capital.loss"] + 1)
    df["work_rate"] = df["hours.per.week"] / (df["age"] + 1)
    df["age_bin"] = pd.cut(df["age"], bins=[0, 30, 50, 100], labels=["young", "mid", "senior"]).astype(str)
    return df

def add_interactions(X):
    return np.hstack([
        X,
        (X[:, [0]] * X[:, [1]]),
        (X[:, [2]] / (X[:, [3]] + 1)),
    ])
