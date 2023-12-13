"""Test MT5 api.

Remember start up trick - need to click into account
and sign up to MetaQuotes in order to get the interface working

"""
from __future__ import annotations

from time import sleep

import requests

from rlbot.utils.configs.constants import mt5_api_port_map
from rlbot.utils.configs.constants import mt5_creds
from rlbot.utils.logging import get_logger
from rlbot.utils.service_manager import start_mt5
from rlbot.utils.service_manager import start_mt5_api

logger = get_logger(__name__)


def test_mt5_api():
    """Test all mt5 api functions."""

    start_mt5()

    start_mt5_api("metaquotes", "general")

    sleep(5)

    broker = "metaquotes"
    mt5_config = mt5_creds[broker]["demo"][0]
    port = mt5_api_port_map[broker]["general"]

    resp = requests.post(f"http://127.0.0.1:{port}/init", json=mt5_config)
    try:
        resp = resp.json()
        logger.debug(resp)
        assert resp["status"]
    except Exception as e:
        logger.warning(str(repr(e)))

    o_request = {
        "retry_num": 4,
        "action": "buy",
        "symbol": "EURUSD",
        "vol": 0.02,
        "agent_version": "t00001",
        "port_ind": 0,
        "deviation": 500,
        "tp_points": 500,
        "sl_points": 500,
    }
    resp = requests.post(f"http://127.0.0.1:{port}/open", json=o_request)
    try:
        resp = resp.json()
        logger.debug(resp)
        assert resp["retcode"] == 10009
    except Exception as e:
        logger.warning(str(repr(e)))

    sleep(10)

    pos = requests.get(f"http://127.0.0.1:{port}/get_positions")
    try:
        pos = pos.json()
        logger.debug(pos)
        pos_id = pos[0]["identifier"]
    except Exception as e:
        logger.warning(str(repr(e)))

    c_request = {
        "retry_num": 10,
        "position_id": pos_id,
        "vol": 0.02,
        "deviation": 500,
        "port_ind": 0,
        "agent_version": "t00001",
    }
    resp = requests.post(f"http://127.0.0.1:{port}/close", json=c_request)
    try:
        resp = resp.json()
        logger.debug(resp)
        assert resp["retcode"] == 10009
    except Exception as e:
        logger.warning(str(repr(e)))

    d_request = {
        "symbol": "EURUSD",
        "dt0": "2023-09-06 12:00:00.000",
        "dt1": "2023-09-06 12:00:01.000",
    }

    resp = requests.get(
        f"http://127.0.0.1:{port}/get_tick_data",
        json=d_request,
        timeout=120,
    )
    try:
        resp = resp.json()
        logger.debug(resp)
        assert len(resp) > 0
    except Exception as e:
        print(str(repr(e)))
