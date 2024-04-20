"""Generate signals.

Generates signals according to an agent version and pushes the json prediction
to redis json. Key to fetch prediction is the agent version. Each new record
currently overwrites the value in redis

TODO test stream or other functionality for redis

TODO wrap in a whole function + add to cli + manage services

"""
from __future__ import annotations

from time import sleep

from rlbot.signals.signal_generator import SignalGenerator
from rlbot.utils.configs.config_builder import load_config
from rlbot.utils.time import get_current_mt5_time
from rlbot.utils.time import mt5_hour_diff
from rlbot.utils.time import wait_till_action_time


def generate_signal(agent_version):
    """Generate a trading signal.

    Spins up processes to extract data from MT5, build features, invoke
    RL model and push the predictions to redis, where the primary key
    is the agent_version.

    Args:
        agent_version (str):
            i.e. 't00001'

    Returns:
        None

    """
    config = load_config(agent_version)

    # initialis signal generator
    sg = SignalGenerator(agent_version)

    hour_diff = mt5_hour_diff()
    now = get_current_mt5_time(hour_diff).strftime("%Y-%m-%d %H:%M:%S.%f")
    sg.pm.now = now
    sg.initialise_data(now)

    while True:
        # sleep so that the data and action is collected at the right interval
        # i.e. if the trade_timeframe = 10s and trade_time_offset = 3s, then
        # sleeps until 13th, 23rd, 33rd, 43rd and 53rd second of each minute
        now = get_current_mt5_time(hour_diff)
        # TODO should wait_till_action_time take now as a string or datetime?
        weekend, sleep_time = wait_till_action_time(
            config.raw_data.trade_timeframe,
            config.raw_data.trade_time_offset,
            now,
        )
        sleep(sleep_time)
        if weekend:
            continue

        # update the now variable with current time
        now = get_current_mt5_time(hour_diff).strftime("%Y-%m-%d %H:%M:%S.%f")
        sg.pm.now = now

        # extract tick data and update features
        sg.update_data(now)

        # predict data - this gets pushed to redis
        _ = sg.predict()
