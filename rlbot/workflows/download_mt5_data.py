"""Download data from mt5.

Start date for all data is 2019-01-01

Todo:
- investigate old / stale tickers
- create code to fill in missing dates (not just latest)
- should longer time series be used?

"""
from __future__ import annotations

import gc
import os
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from glob import glob

import pytz

from rlbot.data.extractor import download_tick_data
from rlbot.utils.configs.constants import mt5_api_port_map
from rlbot.utils.configs.constants import root_dir
from rlbot.utils.logging import get_logger
from rlbot.utils.time import ceil_dt
from rlbot.workflows.service_manager import get_pids
from rlbot.workflows.service_manager import kill_processes
from rlbot.workflows.service_manager import start_all_mt5_apis
from rlbot.workflows.service_manager import stop_mt5

logger = get_logger(__name__)
data_mode = "demo"


def get_most_recent_file(broker, symbol):
    """Searches local save folder for latest tick data file.

    Args:
        broker (str):
            i.e. 'metaquotes' or 'ampglobal'
        symbol (str):
            i.e. 'EURUSD'

    Returns:
        None

    """
    utc_tz = pytz.timezone("Etc/UTC")
    s = symbol.replace("@", "")
    files = list(sorted(glob(f"{root_dir}/data/tick_data/{broker}/{s}/*")))
    # two files are required because we take the second last file as the date
    # last file is often a partial month of data only
    if len(files) < 2:
        dt0 = datetime(2023, 5, 1, tzinfo=utc_tz)
    else:
        last_dt = files[-2].split("/")[-1].split(".")[0].split("_")[-1]
        dt0 = datetime.strptime(last_dt, "%Y-%m-%d")
        dt0 = utc_tz.localize(dt0)
        dt0 = ceil_dt(dt0, "month")
    return dt0


def download_and_save_mt5_tick_data(
    broker,
    symbol,
    dt0,
    dt1,
    data_mode,
    check_api,
    timeout,
):
    """Download and save mt5 tick data.

    Args:
        broker (str):
            i.e. 'metaquotes' or 'ampglobal'
        symbol (str):
            i.e. 'EURUSD'
        dt0 (datetime.datetime(utc))
            start datetime in utc timzone in format "%Y-%m-%d %H:%M:%S.%f"
        dt1 (datetime.datetime(utc))
            end datetime in utc timzone in format "%Y-%m-%d %H:%M:%S.%f"
            #TODO check if this bound is included
        data_mode (str)
            either 'demo' or 'live'
        check_api (bool)
            check mt5 connectivity and attempt to reconnect if not connected

    Returns:
        None

    """
    try:
        df = download_tick_data(broker, symbol, dt0, dt1, data_mode, check_api, timeout)[
            "tick_df"
        ]
        str0 = dt0.strftime("%Y-%m-%d")
        str1 = (dt1 - timedelta(hours=1)).strftime("%Y-%m-%d")
        s = symbol.replace("@", "")
        folder = f"{root_dir}/data/tick_data/{broker}/{s}"
        os.makedirs(folder, exist_ok=True)
        f = f"{folder}/{str0}_{str1}.parquet"
        df.to_parquet(f, engine="pyarrow")
        logger.info(
            (
                f"{broker.ljust(10)} |"
                f" {symbol.ljust(9)}:"
                f" {str(len(df)).rjust(8)} |"
                f" {str0} >> {str1}"
            ),
        )
        df = None
        gc.collect()
        return True
    except Exception as e:
        logger.warning(
            f"{broker.ljust(10)} | {symbol.ljust(9)}: {str(repr(e))}",
        )
        return False


if __name__ == "__main__":
    # start apis
    _ = start_all_mt5_apis()

    data_mode = "demo"
    check_api = True
    timeout = 300

    dl_args = []

    for broker, port_map in mt5_api_port_map.items():
        for symbol, port in port_map.items():
            # general is the port used for other interactions with mt5, i.e. order and
            # getting position
            if symbol != "general":
                dt0 = get_most_recent_file(broker, symbol)
                while dt0 < datetime.now(pytz.timezone("Etc/UTC")):
                    dt1 = ceil_dt(deepcopy(dt0), "month")
                    dl_arg = [broker, symbol, dt0, dt1, data_mode, check_api, timeout]
                    success = download_and_save_mt5_tick_data(*dl_arg)
                    gc.collect()
                    if success:
                        dt0 = ceil_dt(deepcopy(dt0), "month")
                        dt1 = ceil_dt(deepcopy(dt1), "month")

    # kill mt5
    stop_mt5()

    # kill mt5 api process ids
    pids = get_pids("wineserver")
    kill_processes(pids)
    print(f"mt5 apis stopped - process ids {pids} killed")

    # kill wine processes
    pids = get_pids("python.exe")
    kill_processes(pids)
