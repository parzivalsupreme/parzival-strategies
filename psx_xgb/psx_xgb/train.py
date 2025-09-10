import os
import json
import pandas as pd
from xgboost import XGBRegressor

from psx_xgb.data import load_data
from psx_xgb.features import build_features
from psx_xgb.preprocess import clean_data, scale_features
from psx_xgb.modeling import get_default_model
from psx_xgb.eval import evaluate_regression
from psx_xgb.plotting import save_feature_importance


def main():
    os.makedirs('artifacts', exist_ok=True)

    # Load
    df = load_data()

    # Feature engineering
    df = build_features(df)

    # Target
    df['Return_3d'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df = df.dropna()

    # Clean and split
    df = clean_data(df)
    features = [
        'ret_1d','retu_3d','ret_5d','sma_5','sma_20','sma_diff_5_20',
        'ema_10','ema_50','ema_diff_10_50','zscore_20d','rsi_14','macd_hist',
        'tr','atr_14','realized_vol_10d','vol_norm_ret','volume_change',
        'ppo_60d_130d','trix_60d','trix_130d','price_factor'
    ]
    X = df[features]
    y = df['Return_3d']

    X_scaled, scaler = scale_features(X, fit=True)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Model
    model: XGBRegressor = get_default_model()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    # Eval
    metrics = evaluate_regression(model, X_test, y_test)
    print(metrics)

    # Save artifacts
    model.save_model('artifacts/xgb_model.json')
    pd.Series(features).to_json('artifacts/features.json', orient='values')
    import joblib
    joblib.dump(scaler, 'artifacts/scaler.pkl')

    # Plots
    save_feature_importance(model, features, out_path='artifacts/feature_importance.png')


if __name__ == '__main__':
    main()


