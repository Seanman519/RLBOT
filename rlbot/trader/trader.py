"""Trader.

A lot of abstraction required to:
- read multi-timeframe signals from redis
- combine signal
- scale lot size
- manage risk
- trade embargos for macro / session open / close

"""
from __future__ import annotations

from time import sleep

import pandas as pd
import redis
import requests

from rlbot.gym_env.action_processor import build_action_map
from rlbot.gym_env.action_processor import build_pos_arrs
from rlbot.gym_env.action_processor import make_action_labels
from rlbot.utils.configs.config_builder import load_config
from rlbot.utils.configs.constants import mt5_api_port_map
from rlbot.utils.configs.constants import mt5_creds
from rlbot.utils.logging import get_logger

logger = get_logger(__name__)
agent_version = "t00001"
config = load_config(agent_version, enrich_feat_spec=True, is_training=False)

# connect to redis to read rl agent prediction
redis = redis.Redis(host=config.redis.host, port=config.redis.port, decode_responses=True)

# action_map - each index corresponds to a index in the rl agent's prediction
# the columns describe the type of action to take, i.e. long/short or open/close
action_map = build_action_map(config.trader)

# gym representation of positions
gym_portfolio = build_pos_arrs(config.trader)

# transforms index of rl agent's prediction into plain english
action_labels = make_action_labels(config, action_map, gym_portfolio)

# TODO tidy up logic
# load broker information for trading
broker = "metaquotes"
mt5_config = mt5_creds[broker]["demo"]
port = mt5_api_port_map[broker]["general"]

# set trade lot base size - some agent can buy or sell multiple base lots
# TODO change this as part of risk management strategy
# could be time based or macro event based
lot_size = 0.01

# template for open position requests - None indicates it needs to be filled each time
o_request = {
    "retry_num": 4,
    "action": None,
    "symbol": None,
    "vol": None,
    "agent_version": agent_version,
    "port_ind": 0,
    "deviation": 10,
    "tp_points": 100,
    "sl_points": 100,
}

# tempate for close position requests - None indicates it needs to be filled each time
c_request = {
    "retry_num": 10,
    "position_id": None,
    "vol": None,
    "deviation": 10,
    "port_ind": None,
    "agent_version": agent_version,
}


symbols = [x.symbol for x in config.symbol_info]

t0 = ""

while True:
    # read most recent record from redis
    pred = redis.json().get(agent_version, "$")
    t1 = pred[0]["time"]
    # check that the time of current prediction != previous prediction
    if t1 != t0:
        log_msg = ""

        # action index of prediction that takes into account positions
        action_ind = pred[0]["action_pos"]
        # print action
        log_msg += " ".join(action_labels[action_ind].split("_")[:3]).ljust(18)

        # map action to action map to translate prediction into a trading action
        action = action_map[action_ind]
        # portfolio index
        port_ind = action[0]
        # open or close
        # is_open = True if action[2]==1 else False
        # pos_dir
        pos_dir = "buy" if action[1] == 1 else "sell"
        # pos_size
        pos_size = action[3]

        # position
        gym_pos = gym_portfolio[port_ind]
        # first index in gym_portfolio indicates symbol index
        symbol = symbols[int(gym_pos[1])]
        mt5_pos = requests.get(f"http://127.0.0.1:{port}/get_positions").json()

        # print the profit in pips if there are open positions
        profit = 0
        if len(mt5_pos) > 0:
            mt5_pos = pd.DataFrame(mt5_pos)
            for i in mt5_pos.index:
                symbol = mt5_pos.loc[i, "symbol"]
                symbol_index = config.symbol_info_index[symbol]
                pip = config.symbol_info[symbol_index].pip
                # profit in pips
                profit += (
                    mt5_pos.loc[i, "price_current"] - mt5_pos.loc[i, "price_open"]
                ) / pip
        else:
            profit = 0
        log_msg += "profit: " + f"{profit:.1f}".rjust(6)

        # open position
        if action[2] == 1:
            o_request["symbol"] = symbol
            o_request["action"] = pos_dir
            o_request["vol"] = lot_size * pos_size
            resp = requests.post(f"http://127.0.0.1:{port}/open", json=o_request)

        # close position
        elif action[2] == -1:
            a = (mt5_pos["comment"] == agent_version) & (mt5_pos["magic"] == port_ind)
            mt5_pos = mt5_pos[a].iloc[0]
            mt5_pos_id = mt5_pos["identifier"]

            c_request["vol"] = lot_size * pos_size
            c_request["port_ind"] = int(port_ind)
            c_request["position_id"] = int(mt5_pos_id)
            resp = requests.post(f"http://127.0.0.1:{port}/close", json=c_request)

        logger.info(log_msg)
    sleep(pd.to_datetime("now", utc=True).microsecond * 1e-6 + 0.1)
    t0 = t1
