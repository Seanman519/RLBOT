from __future__ import annotations

import pandas as pd

from rlbot.trader.signal_generator import SignalGenerator
from rlbot.utils.configs.config_builder import load_config
from rlbot.utils.time import mt5_hour_diff


hour_diff = mt5_hour_diff()

agent_version = "t00001"
config = load_config(agent_version)

sg = SignalGenerator(agent_version)

now = "2023-09-07 12:00:00.000"
sg.initialise_data(now)


for _ in range(10):
    now = pd.to_datetime(now) + pd.Timedelta("10s")
    sg.update_data(now.strftime("%Y-%m-%d %H:%M:%S.%f"))
    pred = sg.predict()
    print(pred)
