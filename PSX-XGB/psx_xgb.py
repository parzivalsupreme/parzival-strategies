import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/parzivalsupreme/Dataset/refs/heads/main/gold_price.csv"

df = pd.read_csv(url)


# Features and target variable
# Rolling Returns (1d, 3d, 5d)
df["ret_1d"] = df["Close"].pct_change(1)
df["retu_3d"] = df["Close"].pct_change(3)
df["ret_5d"] = df["Close"].pct_change(5)

# SMA/EMA Differences (5/20, 10/50)
df["sma_5"] = df["Close"].rolling(5).mean()
df["sma_20"] = df["Close"].rolling(20).mean()
df["sma_diff_5_20"] = df["sma_5"] - df["sma_20"]

df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
df["ema_diff_10_50"] = df["ema_10"] - df["ema_50"]

# Rolling Z-Score vs 20d MA
rolling_mean = df["Close"].rolling(window=20).mean()
rolling_std = df["Close"].rolling(window=20).std()
df["zscore_20d"] = (df["Close"] - rolling_mean) / rolling_std

# RSI (Relative Strength Index, 14)
delta = df["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)

avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()

rs = avg_gain / avg_loss
df["rsi_14"] = 100 - (100 / (1 + rs))

# MACD Histogram
ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
signal = macd.ewm(span=9, adjust=False).mean()
df["macd_hist"] = macd - signal

# ATR (Average True Range, 14)
df["tr"] = df[["High", "Low", "Close"]].apply(
    lambda x: max(
        x["High"] - x["Low"], abs(x["High"] - x["Close"]), abs(x["Low"] - x["Close"])
    ),
    axis=1,
)
df["atr_14"] = df["tr"].rolling(window=14).mean()

# Realized Volatility (10d)
df["realized_vol_10d"] = df["ret_1d"].rolling(window=10).std() * np.sqrt(252)

# Volatility Normalized Return
df["vol_norm_ret"] = df["ret_1d"] / df["atr_14"]

# Volume Change (Daily)
if "Volume" in df.columns:
    df["volume_change"] = df["Volume"].pct_change(1)

# PPO (60d vs 130d EMA)
ema_60 = df["Close"].ewm(span=60, adjust=False).mean()
ema_130 = df["Close"].ewm(span=130, adjust=False).mean()
df["ppo_60d_130d"] = (ema_60 - ema_130) / ema_130


# TRIX (60d and 130d)
def trix(series, window):
    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change()


df["trix_60d"] = trix(df["Close"], 60)
df["trix_130d"] = trix(df["Close"], 130)

# Price factor
df["price_factor"] = (df["Close"] - df["Close"].mean()) / df["Close"].std()

# Target
df["Return_3d"] = (df["Close"].shift(-3) - df["Close"]) / df["Close"]


# Cleaning
# Drop rows with NaN values
df.dropna(inplace=True)


# Data Cleaning Function
def clean_data(df):
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


# Clean the data
data = clean_data(df)


# Define features & Target
features = [
    # Rolling Returns
    "ret_1d",
    "retu_3d",
    "ret_5d",

    # SMA/EMA Differences
    "sma_5",
    "sma_20",
    "sma_diff_5_20",
    "ema_10",
    "ema_50",
    "ema_diff_10_50",

    # Z-Score
    "zscore_20d",

    # RSI
    "rsi_14",

    # MACD
    "macd_hist",

    # ATR
    "tr",
    "atr_14",

    # Volatility
    "realized_vol_10d",
    "vol_norm_ret",

    # Volume
    "volume_change",

    # PPO
    "ppo_60d_130d",

    # TRIX
    "trix_60d",
    "trix_130d",

    # Price factor
    "price_factor",
]


X = data[features]
y = data["Return_3d"]


# Data transformation
# RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# Final check for any remaining infinite values
X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan)
X_scaled = X_scaled.fillna(0)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

from xgboost import XGBRegressor

model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=0.0,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist'
)

# Train
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# Prediction
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)                # Mean Absolute Error (MAE)

mse = mean_squared_error(y_test, y_pred)                 # Mean Squared Error (MSE)

rmse = np.sqrt(mse)                                      # Root Mean Squared Error (RMSE)

print(mae)
print(mse)
print(rmse)

# Always show a feature importance plot using the fitted model's importances
# Minimal matplotlib-only feature importance plot
fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
ax = fi.plot(kind='barh', figsize=(10, max(4, len(features) * 0.35)))
ax.set_title('Feature Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=220, bbox_inches='tight')
plt.close()

# Tree plotting requires Graphviz; skipping per request to use only matplotlib.

