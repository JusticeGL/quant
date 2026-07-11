from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from alpha_lab.baseline.config import ModelConfig


def make_model(config: ModelConfig, seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        objective=config.objective,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        n_jobs=config.n_jobs,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    config: ModelConfig,
    seed: int,
) -> tuple[LGBMRegressor, np.ndarray]:
    if train.empty or validation.empty:
        raise ValueError("train and validation data must both be non-empty")
    model = make_model(config, seed)
    model.fit(train[feature_names], train["LABEL"])
    prediction = np.asarray(model.predict(validation[feature_names]), dtype=float)
    if prediction.shape != (len(validation),):
        raise RuntimeError("model returned an unexpected prediction shape")
    return model, prediction
