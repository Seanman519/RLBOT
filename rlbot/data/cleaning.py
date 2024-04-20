"""Loads and cleans raw tick data.

Converts to:
- tick
- tick candle
- time candle
- associated time groups so that we can do aggregations
"""
from __future__ import annotations

import os
from datetime import datetime

import polars as pl
from dateutil.relativedelta import relativedelta

from rlbot.data.extractor import download_tick_data


def clean_raw_tick_data(df, tick_time_diff_clip_val):
    """Clean MT5 raw tick data.

    Args:
        df (polars.DataFrame):
            tick data extracted from mt5
        tick_time_diff_clip_val (float):
            a max time to clip the time difference. Necessary because between weekends
            / market close periods, we may see large time differences

    Returns:
        polars.DataFrame:
            clean tick data

    #TODO include real_volume when working on CME / futures data

    """
    df = df.with_columns(((pl.col("bid") + pl.col("ask")) / 2).alias("avg_price"))
    df = df.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
    df = df.with_columns(
        (
            (pl.col("time_msc") - pl.col("time_msc").shift()).cast(
                pl.Int64,
            )
            / 1e9
        )
        .clip(0, tick_time_diff_clip_val)
        .alias("time_diff"),
    )
    df = df.with_columns(pl.col("flags") % 128)
    df = df.select(
        ["bid", "ask", "time_msc", "avg_price", "spread", "time_diff", "flags"],
    )
    return df


def fill_trade_interval(df, trade_timeframe, fill_strategy):
    """Fill missing timestamps.

    Fill in missing trade_intervals, during no trade periods and
    low volume periods, whilst excluding weekends, christmas and new years day

    #TODO date fill based on mt5 local time, change to utc for other
    brokers / platforms

    Args:
        df (polars.DataFrame):
            dataframe with column "time_msc"
        trade_timeframe (str):
            trade interval in polars format i.e. '10s'
        fill_strategy (str):
            see polars documentation, i.e. 'forward','zero', etc.

    Returns:
        pl.Dataframe

    """
    df = (
        pl.DataFrame(
            {
                "time_msc": pl.date_range(
                    start=df["time_msc"].min(),
                    end=df["time_msc"].max(),
                    interval=trade_timeframe,
                    eager=True,
                    time_unit="ns",
                ),
            },
        )
        .join(df, on="time_msc", how="outer")
        .fill_nan(None)
        .fill_null(strategy=fill_strategy)
        # exclude  weekends
        .filter(pl.col("time_msc").dt.weekday() <= 5)
        # exclude christmas
        .filter(
            ~(
                (pl.col("time_msc").dt.day() == 25)
                & (pl.col("time_msc").dt.month() == 12)
            ),
        )
        # exclude new year
        .filter(
            ~((pl.col("time_msc").dt.day() == 1) & (pl.col("time_msc").dt.month() == 1)),
        )
    )
    return df


def group_tick_data_by_time(config, feat_group_ind, tick_df, mode="train", lazy=False):
    """Groups tick data by time.

    Args:
        config:
            Config class from the config_builder.py file
        feat_group_ind (int):
            index of feat group ind
        tick_df (polars.DataFrame):
            tick data
        mode (str):
            possible values include 'train' or anything else
        lazy (bool):
            if True, use lazy polars dataframe

    If the first tick timestamp is 2023-04-28 00:18:01.005, then the
    first group will be after that, i.e. 2023-04-28 00:18:03.001

    """
    feature_timeframe = config.features[feat_group_ind].timeframe
    trade_timeframe = config.raw_data.trade_timeframe

    tick_df = tick_df.with_columns(
        pl.col("time_msc").dt.offset_by(by="-" + config.raw_data.trade_time_offset),
    )

    if lazy:
        df_group = (
            tick_df.lazy()
            .set_sorted("time_msc")
            .group_by_dynamic(
                "time_msc",
                every=trade_timeframe if mode == "train" else feature_timeframe,
                period=feature_timeframe,
                closed="left",
                start_by="window" if mode == "train" else "datapoint",
                check_sorted=False,
            )
        )
    else:
        df_group = tick_df.set_sorted("time_msc").group_by_dynamic(
            "time_msc",
            every=trade_timeframe if mode == "train" else feature_timeframe,
            period=feature_timeframe,
            closed="left",
            start_by="window" if mode == "train" else "datapoint",
            check_sorted=False,
        )
    return df_group


def load_raw_tick_data(config, broker, symbol, dt, df=None):
    """Load raw tick data.

    #TODO clean up logic
    #TODO convert all pandas to polars

    Load tick data for 3 different use cases , as differentiated by dt
    if dt == 'update':
        tick data that is used to update an already existing db, used to incrementally
        update tick data so that the agent can learn from the most recent data
    elif dt == 'inference':
        df, a pandas dataframe of tick data which can be downloaded via the MT5 api
        must also be passed in. This is used at inference time at each timestep
    else:
        clean start where we initialise and populate the database with features defined
        by the 'feature_config.py' file in each agent folder. dt is the format '%Y-%m-01'
        to indicate the start of the month of tick data. Note:
            - we also load the previous month of data because some features require 3-4
            hours of tick data, e.g. features for 1 May 2023 00:00:00 may require data
            from 28 April 23:00:00

    Args:
        config (pydantic.BaseModel):
            as defined in the 'agent_config.py'
        symbol (str):
            trading instrument, i.e. EURUSD
        dt (str):
            monthly dates in the format '%y-%m-01' or 'inference' or 'update'
        df (pd.DataFrame | None):
            only provide a pandas dataframe of tick data when dt = 'inference'

    Returns:
        pl.DataFrame:
            polars dataframe of cleaned tick data

    """
    if dt == "update":
        local_tick_data_dir = f"{config.paths.update_tick_data_dir}/{broker}"
        os.makedirs(local_tick_data_dir, exist_ok=True)
        local_f = f"{local_tick_data_dir}/{symbol}.parquet"
        df = pl.read_parquet(local_f, use_pyarrow=True)
        df = clean_raw_tick_data(df, config.raw_data.tick_time_diff_clip_val)
        df = df.select(
            [
                pl.col("bid").cast(pl.Float32),
                pl.col("ask").cast(pl.Float32),
                pl.col("time_msc").dt.cast_time_unit("ns"),
                pl.col("avg_price").cast(pl.Float32),
                pl.col("spread").cast(pl.Float32),
                pl.col("time_diff").cast(pl.Float32),
                pl.col("flags").cast(pl.Int32),
            ],
        )
        return df
    elif dt == "inference":
        df = pl.from_pandas(df)
        df = clean_raw_tick_data(df, config.raw_data.tick_time_diff_clip_val)
        df = df.select(
            [
                pl.col("bid").cast(pl.Float32),
                pl.col("ask").cast(pl.Float32),
                pl.col("time_msc").dt.cast_time_unit("ns"),
                pl.col("avg_price").cast(pl.Float32),
                pl.col("spread").cast(pl.Float32),
                pl.col("time_diff").cast(pl.Float32),
                pl.col("flags").cast(pl.Int32),
            ],
        )
        return df
    else:
        # git the current file name
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
        dt1 = dt + relativedelta(months=1) - relativedelta(days=1)
        curr_f = f"{dt.strftime('%Y-%m-%d')}_{dt1.strftime('%Y-%m-%d')}.parquet"

        # try to read the tail end of the previous file
        prev_f = None
        prev_dt = dt - relativedelta(months=1)
        prev_dt1 = dt - relativedelta(days=1)
        if prev_dt1.year >= 2019:
            prev_f = (
                f"{prev_dt.strftime('%Y-%m-%d')}_{prev_dt1.strftime('%Y-%m-%d')}.parquet"
            )

        local_tick_data_dir = f"{config.paths.tick_data_dir}/{broker}/{symbol}"
        os.makedirs(local_tick_data_dir, exist_ok=True)

        tick_df = []

        for f in [prev_f, curr_f]:
            if f is not None:
                local_f = f"{local_tick_data_dir}/{f}"

                # TODO clean this logic, make polars instead of pandas
                if not os.path.exists(local_f):
                    if prev_f == f:
                        df = download_tick_data(
                            broker,
                            symbol,
                            prev_dt,
                            prev_dt1 + relativedelta(days=1),
                            data_mode=config.raw_data.data_mode,
                        )["tick_df"]
                    elif curr_f == f:
                        df = download_tick_data(
                            broker,
                            symbol,
                            dt,
                            dt1 + relativedelta(days=1),
                            data_mode=config.raw_data.data_mode,
                        )["tick_df"]
                    df.to_parquet(local_f, engine="pyarrow")
                    df = pl.from_pandas(df)
                else:
                    df = pl.read_parquet(local_f, use_pyarrow=True)

                if f == prev_f:
                    df = df.filter(
                        pl.col("time_msc")
                        >= pl.col("time_msc").max().dt.offset_by("-1d"),
                    )
                df = clean_raw_tick_data(df, config.raw_data.tick_time_diff_clip_val)
                tick_df.append(
                    df.select(
                        [
                            pl.col("bid").cast(pl.Float32),
                            pl.col("ask").cast(pl.Float32),
                            pl.col("time_msc").dt.cast_time_unit("ns"),
                            pl.col("avg_price").cast(pl.Float32),
                            pl.col("spread").cast(pl.Float32),
                            pl.col("time_diff").cast(pl.Float32),
                            pl.col("flags").cast(pl.Int32),
                        ],
                    ),
                )
        tick_df = pl.concat(tick_df, how="vertical")
        return tick_df


def get_trade_price(config, broker, symbol, dt):
    """Get trade price.

    For each trade time interval, get the bid and ask price for the next X seconds

    Args:
        config (pydantic.BaseModel):
            defined in agent_config.py
        symbol (str):
            trading isntrument, i.e. EURUSD
        dt (str):
            monthly dates in the format '%Y-%m-01'

    Returns:
        None

    """
    trade_timeframe = config.raw_data.trade_timeframe
    tick_df = load_raw_tick_data(config, broker, symbol, dt)

    tick_df = tick_df.with_columns(
        pl.col("time_msc").dt.offset_by(by="-" + config.raw_data.trade_time_offset),
    )
    df_group = tick_df.set_sorted("time_msc").groupby_dynamic(
        "time_msc",
        every=trade_timeframe,
        period=config.raw_data.max_inf_time_s,
        closed="left",
        # offset=config.raw_data.trade_time_offset,
        check_sorted=False,
    )

    df = df_group.agg(
        [
            pl.col("bid").min().alias("min_bid"),
            pl.col("bid").max().alias("max_bid"),
            pl.col("ask").min().alias("min_ask"),
            pl.col("ask").max().alias("max_ask"),
        ],
    )
    df = df.with_columns(
        pl.col("time_msc").dt.offset_by(by=config.raw_data.trade_time_offset),
    )
    df = fill_trade_interval(df, trade_timeframe, "forward")

    save_dir = f"{config.paths.feature_dir}/trade_price/{broker}/{symbol}"
    os.makedirs(save_dir, exist_ok=True)
    df.write_parquet(f"{save_dir}/{dt}.parquet", use_pyarrow=True)
