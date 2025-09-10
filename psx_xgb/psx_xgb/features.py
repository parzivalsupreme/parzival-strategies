import numpy as np
import pandas as pd


def trix(series: pd.Series, window: int) -> pd.Series:
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Returns
    df["ret_1d"] = df["Close"].pct_change(1)
    df["retu_3d"] = df["Close"].pct_change(3)
    df["ret_5d"] = df["Close"].pct_change(5)

    # SMA/EMA
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_diff_5_20"] = df["sma_5"] - df["sma_20"]

    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ema_diff_10_50"] = df["ema_10"] - df["ema_50"]

    # Z-score
    rolling_mean = df["Close"].rolling(window=20).mean()
    rolling_std = df["Close"].rolling(window=20).std()
    df["zscore_20d"] = (df["Close"] - rolling_mean) / rolling_std

    # RSI
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD hist
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - signal

    # ATR
    df["tr"] = df[["High", "Low", "Close"]].apply(
        lambda x: max(
            x["High"] - x["Low"], abs(x["High"] - x["Close"]), abs(x["Low"] - x["Close"])
        ),
        axis=1,
    )
    df["atr_14"] = df["tr"].rolling(window=14).mean()

    # Volatility
    df["realized_vol_10d"] = df["ret_1d"].rolling(window=10).std() * np.sqrt(252)
    df["vol_norm_ret"] = df["ret_1d"] / df["atr_14"]

    # Volume
    if "Volume" in df.columns:
        df["volume_change"] = df["Volume"].pct_change(1)

    # PPO
    ema_60 = df["Close"].ewm(span=60, adjust=False).mean()
    ema_130 = df["Close"].ewm(span=130, adjust=False).mean()
    df["ppo_60d_130d"] = (ema_60 - ema_130) / ema_130

    # TRIX
    df["trix_60d"] = trix(df["Close"], 60)
    df["trix_130d"] = trix(df["Close"], 130)

    # Price factor
    df["price_factor"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()

    return df


