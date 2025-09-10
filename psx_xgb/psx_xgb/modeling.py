from xgboost import XGBRegressor


def get_default_model() -> XGBRegressor:
    return XGBRegressor(
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


