from __future__ import annotations

from pathlib import Path

import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.data import D


def alpha158_definition() -> tuple[list[str], list[str]]:
    """Return the feature expressions from the pinned Qlib Alpha158 handler."""
    handler = object.__new__(Alpha158)
    expressions, names = handler.get_feature_config()
    expression_list = [str(value) for value in expressions]
    name_list = [str(value) for value in names]
    if len(expression_list) != 158 or len(name_list) != 158:
        raise RuntimeError(
            "pinned Qlib Alpha158 contract changed: "
            f"{len(expression_list)} expressions/{len(name_list)} names"
        )
    if len(set(name_list)) != 158:
        raise RuntimeError("Alpha158 feature names are not unique")
    return expression_list, name_list


def load_alpha158_dataset(
    provider_uri: Path,
    *,
    start_time: str,
    end_time: str,
    label_expression: str,
) -> pd.DataFrame:
    qlib.init(provider_uri=str(provider_uri), region=REG_CN)
    expressions, names = alpha158_definition()
    fields = [*expressions, label_expression]
    columns = [*names, "LABEL"]
    frame = D.features(
        D.instruments("all"),
        fields,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    frame.columns = columns
    result = frame.reset_index()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    return result.sort_values(["datetime", "instrument"], kind="stable").reset_index(
        drop=True
    )
