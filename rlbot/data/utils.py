"""Data utils.

Random function that need to find a better place to belong in

"""
from __future__ import annotations

import pandas as pd
import polars as pl

from rlbot.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_dir(config, feat_group_ind, feat_ind):
    """Get feature dir.

    Gets location of where each raw and scaled feature is stored.

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        feat_group_ind (int):
            index of feature group
        feat_ind (int):
            index of feature within its feature group

    Returns:
        str:
            path
    """
    feat_group = config.features[feat_group_ind]
    fc = feat_group.simple_features[feat_ind]
    feature_dir = (
        f"{config.paths.feature_dir}"
        f"/{feat_group.index}_{feat_group.timeframe}"
        f"/{fc.index}_{fc.name}"
    )
    return feature_dir


def split_timeframe(timeframe):
    """Split timeframe.

    #TODO make it work for pandas and polars and transalte into a common format
    #TODO make it work for float becuase polars allows for float timeframes

    Separates timeframe so we can perform time operations

    Args:
        timeframe (str):
            for example 10s or 1h

    Returns:
        int
            integer value that applies to the unit of time
        str
            unit of time as a string

    """
    num = int("".join([s for s in timeframe if s.isdigit()]))
    unit = "".join([s for s in timeframe if not s.isdigit()])
    return num, unit


def tick_list_to_polars_df(df):
    """List of polars for tick data."""
    df = pl.DataFrame(df)
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.to_datetime().cast(pl.Datetime(time_unit="ns")),
            )
    return df


def tick_polars_df_to_list(df):
    """Polars to list for tick data."""
    df = df.to_pandas()
    for col in df.columns:
        # hacky method of checking whether column is of datetime64[ns]
        if df[col].dtype.str[1] == "M":
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return df.to_dict(orient="records")


def update_tick_data(data, new_data):
    """Update tick data.

    Args:
        data (dict[pd.DataFrame]):
            dictionary of dataframes with tick data
        new_data (dict[pd.DataFrame]):
            dictionary if dataframes with tick data

    Returns:
        dict[pd.DataFrame]
            up to date tick data

    """
    for broker_symbol in data.keys():
        df = data[broker_symbol]
        df = df[df["time_msc"] < df["time_msc"].iloc[-1].replace(microsecond=0)]
        # df = df[df["time_msc"] < now.replace(microsecond=0).replace(tzinfo=None)]
        df = pd.concat([df, new_data[broker_symbol]], axis=0)
        hours = 24
        if (df["time_msc"].max() - pd.Timedelta(hours=24)).dayofweek > 4:
            hours += 48
        df = df[df["time_msc"] > (df["time_msc"].max() - pd.Timedelta(hours=hours))]
        df.reset_index(inplace=True, drop=True)
        data[broker_symbol] = df
    return data
