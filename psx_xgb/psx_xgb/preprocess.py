import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        df[column] = df[column].fillna(df[column].median())
    return df


def scale_features(X: pd.DataFrame, fit: bool = True, scaler: RobustScaler | None = None):
    if scaler is None:
        scaler = RobustScaler()
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler


