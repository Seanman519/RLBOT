from __future__ import annotations

import _pickle as cPickle

import pandas as pd
import requests

from releat.utils.configs.constants import mt5_api_port_map
from releat.utils.configs.constants import mt5_creds
from releat.utils.logging import get_logger
from releat.workflows.service_manager import start_mt5_api


logger = get_logger(__name__)


def download_tick_data(broker, symbol, dt0, dt1, data_mode, check_api=True, timeout=60):
    """Download tick data.

    Downloads tick data between two datetime, dt0 and t1

    #TODO make work for multi-broker

    Args:
        broker (str):
            broker as string
        symbol (str):
            trading instrument, e.g. EURUSD
        dt0 (datetime.datetime):
            starting datetime
        dt1 (datetime.datetime):
            ending datetime - non inclusive i think
        data_mode (str):
            either 'live' or 'demo'
        check_api (bool):
            True or False
        timeout (int):
            timeout of api in seconds

    """
    port = mt5_api_port_map[broker][symbol]
    if check_api:
        resp = requests.get(f"http://127.0.0.1:2000/healthcheck")
        if resp.status_code != 200:
            logger.warning(f"Connection error to port 2000. Initializing...")
            start_mt5_api(broker, symbol)
            resp = requests.get(f"http://127.0.0.1:2000/healthcheck")
            assert resp.stats_code == 200, f"http://127.0.0.1:2000 failed to initialize"

        mt5_config = mt5_creds[broker][data_mode]
        resp = requests.post(f"http://127.0.0.1:2000/init", json=mt5_config)
        logger.info(f"Connection success to port 2000")

    dt0 = dt0.strftime("%Y-%m-%d %H:%M:%S.%f")
    dt1 = dt1.strftime("%Y-%m-%d %H:%M:%S.%f")

    d_request = {
        "symbol": symbol,
        "dt0": dt0,
        "dt1": dt1,
    }

    resp = requests.get(
        f"http://127.0.0.1:2000/get_tick_data",
        json=d_request,
        timeout=timeout,
    )

    resp = cPickle.loads(resp.content)

    if "error" in resp:
        logger.warning(
            f"{broker} {symbol} {dt0} >> {dt1} No ticks {resp}",
        )
        return {
            "broker": broker,
            "symbol": symbol,
            **resp,
        }
    else:
        tick_df = resp["data"]
        tick_df["time_msc"] = pd.to_datetime(tick_df["time_msc"], unit="ms")
        logger.info(
            f"{broker} {symbol} {dt0} >> {dt1} {str(len(tick_df)).rjust(8)} ticks",
        )
        return {
            "broker": broker,
            "symbol": symbol,
            "tick_df": tick_df,
        }
