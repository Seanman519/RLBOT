"""Build data sets given a configuration in real time or batch.

Build data for:
- training data
- updating db
- inference

"""
from __future__ import annotations

import json
import os
from glob import glob

import aerospike
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from rlbot.connectors.aerospike import get_records_in_aerospike
from rlbot.connectors.aerospike import search_aerospike_for_dt
from rlbot.data.cleaning import fill_trade_interval
from rlbot.data.cleaning import get_trade_price
from rlbot.data.cleaning import group_tick_data_by_time
from rlbot.data.cleaning import load_raw_tick_data
from rlbot.data.simple.stats import calc_gradient_feature
from rlbot.data.simple.stats import calc_inflection_feature
from rlbot.data.simple.stats import calc_peak_trough_gradient_feature
from rlbot.data.simple.stats import get_max
from rlbot.data.simple.stats import get_mean
from rlbot.data.simple.stats import get_min
from rlbot.data.simple.stats import get_skew
from rlbot.data.simple.stats import one_hot_fx_flag
from rlbot.data.transformers import apply_transform
from rlbot.data.transformers import enrich_transform_config
from rlbot.data.transformers import get_transform_params_for_all_features
from rlbot.data.utils import get_feature_dir
from rlbot.utils.configs.constants import trading_instruments
from rlbot.utils.logging import get_logger

logger = get_logger(__name__)


def make_feature(df_group, config, feat_group_ind, feat_ind, fname=None, mode="build"):
    """Make a feature.

    Makes one simple feature for a series of tick data groups. These groups are
    tick data that have been grouped by the feature timeframe. The difference between
    the 2 modes of building a feature are:
        inference:
            - does not need to be saved locally
        other:
            - is saved locally so that data can be easily accessed / rebuilt

    #TODO add checks to ensure fname is unique
    #TODO convert to feature factory class

    Args:
        df_group (pl.GroupBy):
            polars dataframe of tick data that has been grouped by the feature timeframe
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        feat_group_ind (int):
            index of the feature group
        feat_ind (int):
            index of the feature within the feature group
        fname (str | None):
            name of the feature file - feature files get saved locally before being
            uploaded to db, enabling faster rebuilds. Not necessary for inference
        mode (str):
            either 'inference' or some other string value


    """
    feat_group = config.features[feat_group_ind]
    fc = feat_group.simple_features[feat_ind]
    feature_timeframe = fc.timeframe
    trade_timeframe = config.raw_data.trade_timeframe
    pip = config.symbol_info[config.symbol_info_index[fc.symbol]].pip

    save_dir = get_feature_dir(config, feat_group_ind, feat_ind)
    save_dir = f"{save_dir}/raw_data"
    os.makedirs(save_dir, exist_ok=True)

    save_f = f"{save_dir}/{fname}.parquet"
    # if os.path.exists(save_f):
    #     return

    match fc.name:
        case "differencing":
            df = get_mean(df_group, fc)
        case "one_hot":
            df = one_hot_fx_flag(df_group, fc)
        case "mean":
            df = get_mean(df_group, fc)
        case "min":
            df = get_min(df_group, fc)
        case "max":
            df = get_max(df_group, fc)
        case "skew":
            df = get_skew(df_group, fc, pip)
        case "grad":
            df = calc_gradient_feature(df_group, fc, pip)
        case "grad_with_peak_trends":
            df = calc_peak_trough_gradient_feature(df_group, fc, pip)
        case "inflection":
            df = calc_inflection_feature(df_group, fc, pip)

    df = df.with_columns(pl.col("time_msc").dt.cast_time_unit("ns"))

    df = df.with_columns(pl.col("time_msc").dt.offset_by(feature_timeframe)).with_columns(
        pl.col("time_msc").dt.offset_by(config.raw_data.trade_time_offset),
    )
    df = fill_trade_interval(df, trade_timeframe, fc.fillna)

    if mode == "inference":
        return df
    else:
        # save as raw data
        save_dir = get_feature_dir(config, feat_group_ind, feat_ind)
        save_dir = f"{save_dir}/raw_data"
        os.makedirs(save_dir, exist_ok=True)

        save_f = f"{save_dir}/{fname}.parquet"

        df = df.filter(
            pl.col("time_msc")
            <= pl.col("time_msc").max().dt.offset_by("-" + feature_timeframe),
        ).filter(
            pl.col("time_msc")
            >= pl.col("time_msc").min().dt.offset_by(feature_timeframe),
        )
        df.write_parquet(save_f, use_pyarrow=True)

        return df


def build_features_by_dt(config, dts):
    """Build features by dt.

    #TODO improve parallelisation?

    When the dataset is initially being built, features are built month by month to
    conserve RAM

    Args:
        config (pydantic.BaseModel):
            defined in 'agent_config.py'
        dts (List(str)):
            list of dates in the format (%Y-%m-01')

    Returns:
        None

    """
    for dt in tqdm(list(sorted(dts))):
        for feat_group_ind in range(len(config.features)):
            feat_group = config.features[feat_group_ind]
            prev_symbol = ""
            for feat_ind in range(len(feat_group.simple_features)):
                fc = feat_group.simple_features[feat_ind]
                symbol = fc.symbol
                broker = fc.broker
                if symbol != prev_symbol:
                    tick_df = load_raw_tick_data(config, broker, symbol, dt)
                    if fc.name in ["grad", "fft"]:
                        df_group = group_tick_data_by_time(
                            config,
                            feat_group_ind,
                            tick_df,
                            lazy=True,
                        )
                    else:
                        df_group = group_tick_data_by_time(
                            config,
                            feat_group_ind,
                            tick_df,
                        )

                logger.info(f"{dt} - making feature {feat_group_ind}-{feat_ind}")

                _ = make_feature(
                    df_group,
                    config,
                    feat_group_ind,
                    feat_ind,
                    fname=dt,
                    mode="build",
                )


def upload_trade_data(config, client, dt, start_val=None):
    """Upload trade data.

    # TODO make it work better for multi symbol
    # TODO decouple from aerospike

    Upload trade data from local file to database.

    Args:
        config (pydantic.BaseModel):
            as defined in agent_config.py
        client (aerospike.client):
            client for uploading data
        dt (pd.DateTime):
            pandas datetime of first record
        start_val (int | None):
            the table index to start searching for the index that matches the date
            used to speed up the index search

    Returns:
        None

    """
    dfs = None
    for si in config.symbol_info:
        symbol = si.symbol
        broker = si.broker
        save_dir = f"{config.paths.feature_dir}/trade_price/{broker}/{symbol}"
        df = pl.read_parquet(f"{save_dir}/{dt}.parquet", use_pyarrow=True)
        # TODO add in timezone change here!
        df = df.with_columns(pl.col("time_msc").cast(pl.Utf8))
        if dfs is None:
            dfs = df.clone()
        else:
            dfs = dfs.join(df, on="time_msc", how="inner")

    df = dfs.to_numpy()
    dt = pd.to_datetime(df[0, 0][:19])
    dt = int(dt.timestamp())
    ind_offset = search_aerospike_for_dt(config, client, dt, start_val)

    # cols = [x for x in df.columns if x != 'time_msc']
    for i in tqdm(range(len(df))):
        key = (
            config.aerospike.namespace,
            config.aerospike.set_name,
            int(i + ind_offset),
        )

        dt = df[i, 0][:19]
        date_arr = [
            np.round(float(dt[11:13]) / 23, decimals=6),
            np.round(float(dt[14:16]) / 59, decimals=6),
            np.round(float(dt[17:19]) / 59, decimals=6),
        ]

        dt = pd.to_datetime(dt)
        dt = int(dt.timestamp())
        # try:
        #     client.remove(key)
        # except:
        #     pass

        data = df[i, 1:]
        data = np.round(data.astype(float), decimals=6).tolist()

        bins = {
            "date": dt,
            "trade_price": data,
            "date_arr": date_arr,
        }
        client.put(key, bins)

    dt0 = df[0, 0][:19]
    dt1 = df[-1, 0][:19]
    logger.info(f"trade data updated for {dt0} - {dt1}")


def upload_feature_group(config, client, feat_group_ind, dt, start_val):
    """Upload feature group.

    Aggregates features from local storage and uploads feature group to database.
    Note one feature group = one timeframe (for now), i.e. you can't have two
    different feature groups that are of 10m timeframes because the key that is
    used for each record is the timeframe

    #TODO decouple from aerospike

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        client (aerospike.Client):
            client for reading and inserting database records into aerospike
        feat_group_ind (int):
            index of the feature group
        dt (pd.DateTime):
            datetime of the first record in the feature group
        start_val (int):
            table index that we start searching the date from - this just speeds
            up the search

    """
    feat_group = config.features[feat_group_ind]
    dfs = None
    for feat_ind in range(len(feat_group.simple_features)):
        scaled_obs_dir = get_feature_dir(config, feat_group_ind, feat_ind)
        f = f"{scaled_obs_dir}/raw_data/{dt}.parquet"

        # TODO add in timezone change here
        df = pd.read_parquet(f).set_index("time_msc")

        fc = config.features[feat_group_ind].simple_features[feat_ind]

        if fc.name != "differencing":
            for t_ind in range(len(fc.transforms)):
                fc = enrich_transform_config(
                    config,
                    feat_group_ind,
                    feat_ind,
                    t_ind,
                    feat_shape=df.shape,
                )

            feats = df.values.astype("float32")
            for tc in fc.transforms:
                feats = apply_transform(feats, tc)
            df = pd.DataFrame(feats, index=df.index, columns=df.columns)

        if dfs is None:
            dfs = df
        else:
            dfs = pd.concat([dfs, df], axis=1, join="inner")

    assert dfs.isnull().sum().sum() == 0

    df = dfs

    key = (
        config.aerospike.namespace,
        config.aerospike.set_name,
        0,
    )
    (_, _, old_bins) = client.get(key)
    df = df[df.index > pd.to_datetime(old_bins["date"], unit="s")]

    # dt = df.index[0].strftime("%Y-%m-%d %H:%M:%S")
    # dt = datetime.datetime.strptime(dt,'%Y-%m-%d %H:%M:%S')
    dt = int(df.index[0].timestamp())
    ind_offset = search_aerospike_for_dt(config, client, dt, start_val)

    for i in tqdm(range(len(df))):
        # dt = df.index[i].strftime("%Y-%m-%d %H:%M:%S")
        dt = int(df.index[i].timestamp())
        arr = np.round(df.iloc[i].values.astype(float), decimals=6).tolist()

        key = (
            config.aerospike.namespace,
            config.aerospike.set_name,
            int(i + ind_offset),
        )

        bins = {
            "date": dt,
            str(feat_group_ind): arr,
        }

        _, meta = client.exists(key)
        if meta is not None:
            (_, _, old_bins) = client.get(key)

            if "date" in old_bins:
                assert old_bins["date"] == dt, (
                    f"current date {old_bins['date']} ",
                    "new data dates: ",
                    df.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                    int(df.index[i].timestamp()),
                )

            client.put(key, bins)


def upload_scaled_obs(config, client, dt, ind):
    """Upload all scaled obs."""
    for feat_group_ind in range(len(config.features)):
        logger.info(f"uploading feature group {feat_group_ind} to db")
        _ = upload_feature_group(config, client, feat_group_ind, dt, ind)


def update_gym_env_hparam(config, client, overwrite_max_samples=False):
    """Update hyperparameters.

    Gym environment configs and hyperparameters are uploaded to database so that
    we can dynamically change them without restarting training.

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        client (aerospike.Client):
            aerospike client used to download and insert records

    Returns:
        None

    """
    key = (
        config.aerospike.namespace,
        config.aerospike.set_name + "_hparams",
        "gym_env_configs",
    )

    bins = config.gym_env.dict()
    for k in bins:
        assert len(k) <= 14, f"bin name {k} > 14 characters"

    incomplete_record = True
    max_data_ind = get_records_in_aerospike(config, client)
    while incomplete_record:
        k = (
            config.aerospike.namespace,
            config.aerospike.set_name,
            max_data_ind,
        )
        _, _, old_bins = client.get(k)

        complete = True
        for i in range(len(config.features)):
            complete = complete & (str(i) in old_bins)
        if complete:
            incomplete_record = False
        else:
            max_data_ind -= 1

    bins["max_data_ind"] = max_data_ind

    _, meta = client.exists(key)
    if (meta is None) | (overwrite_max_samples):
        bins["max_samples"] = bins["rec_num"]
    else:
        (_, _, old_bins) = client.get(key, bins)
        bins["max_samples"] = old_bins["max_samples"]

    client.put(key, bins)
    logger.info(f"gym config and hparams updated - total recs: {max_data_ind}")


def populate_train_data(config, mode="update"):
    """Populate train data.

    From downloading tick data to features loaded in database. There are two modes:
        initialise:
            build database from scratch
        update:
            add new data to existing database

    Args:
        config (pydantic.BaseModel):
            as defined in 'agent_config.py'
        mode (str):
            values can either be 'initialise' or 'update'

    Returns:
        None

    """
    client = aerospike.client(config.aerospike.connection).connect()

    if mode == "update":
        dts = ["update"]
        dt = dts[0]

        for feat_group_ind in range(len(config.features)):
            feat_group = config.features[feat_group_ind]
            prev_symbol = ""

            for feat_ind in range(len(feat_group.simple_features)):
                fc = feat_group.simple_features[feat_ind]
                symbol = fc.symbol
                broker = fc.broker
                if symbol != prev_symbol:
                    tick_df = load_raw_tick_data(config, broker, symbol, dt)
                    df_group = group_tick_data_by_time(config, feat_group_ind, tick_df)

                _ = make_feature(
                    df_group,
                    config,
                    feat_group_ind,
                    feat_ind,
                    fname=dt,
                    mode="build",
                )

    elif mode == "initialise":
        dts = config.raw_data.tick_file_dates
        if dts is None:
            dts = []
            for broker, symbols in trading_instruments.items():
                for symbol in symbols:
                    files = list(
                        glob(f"{config.paths.tick_data_dir}/{broker}/{symbol}/*"),
                    )
                    dts += [x.split("/")[-1].split("_")[0] for x in files]
            dts = list(set(dts))
        dts = list(sorted(dts))
        _ = build_features_by_dt(config, dts)
        _ = get_transform_params_for_all_features(config)

    for i in tqdm(range(len(dts))):
        dt = dts[i]

        for symbol_info in config.symbol_info:
            _ = get_trade_price(config, symbol_info.broker, symbol_info.symbol, dt)

        start_val = None
        _ = upload_trade_data(config, client, dt, start_val)

        _ = upload_scaled_obs(config, client, dt, start_val)

    _ = update_gym_env_hparam(config, client)

    max_data_ind = get_records_in_aerospike(config, client)
    key = (
        config.aerospike.namespace,
        config.aerospike.set_name,
        int(max_data_ind),
    )

    (_, _, bins) = client.get(key)
    with open(
        f"{config.paths.update_tick_data_dir}/update_tick_data_info.json",
        "w",
    ) as f:
        json.dump(bins, f)
