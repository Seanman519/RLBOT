"""Portfolio Manager."""
from __future__ import annotations

import pandas as pd
import requests

from rlbot.gym_env.action_processor import build_pos_arrs
from rlbot.utils.configs.config_builder import load_config


class PortfolioManager:
    """Portfolio Manager.

    Manages open and closed positions, translating between the mt5 and rl gym
    representation.

    """

    def __init__(self, agent_version):
        """Init."""
        self.agent_version = agent_version
        self.tick_data = None
        self.feature_data = None
        self.gym_data = None
        self.config = load_config(agent_version, enrich_feat_spec=True, is_training=False)
        self.gym_portfolio = build_pos_arrs(self.config.trader)
        self.gym_portfolio_hedge = build_pos_arrs(self.config.trader)
        self.trade_lot = self.config.trader.lot
        self.total_mt5_portfolio = None
        # TODO parameteris this :2000 is for EURUSD only
        self.data_api = "http://127.0.0.1:2000"

    def update_gym_portfolio(self, gym_portfolio, mt5_portfolio):
        """Update gym portfolio.

        Using data from the mt5 portfolio, update the gym portfolio.

        Args:
            gym_portfolio (np.array)
                n x 13 array, where n depends on the available trading instruements
            mt5_portfolio (np.array)

        Returns
            np.array:
                updated gym portfolio

        """
        # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
        # pos_dir, open_time, curr_time, hold_time, open_price, curr_price, curr_val
        # TODO make all time variables in ms instead of timestam[s so we can njit
        # pos_size
        gym_portfolio[:, 5] = mt5_portfolio["volume"].values / self.trade_lot
        # pos_direction
        gym_portfolio[:, 6] = (1 - mt5_portfolio["type"] * 2).values
        # hold_time
        gym_portfolio[:, 9] = (
            (
                pd.to_datetime(self.now).tz_localize(None)
                - mt5_portfolio["time_msc"].dt.tz_localize(None)
            ).dt.total_seconds()
            // 10
        ).values
        # current val
        gym_portfolio[:, 12] = (
            (mt5_portfolio["price_current"] - mt5_portfolio["price_open"]).values
            / gym_portfolio[:, 2]
            * gym_portfolio[:, 6]
        )
        return gym_portfolio

    def get_mt5_portfolio(self):
        """Update MT5 portfolio."""
        total_portfolio = requests.get(f"{self.data_api}/get_positions").json()
        self.total_mt5_portfolio = None
        if len(total_portfolio) > 0:
            self.total_mt5_portfolio = pd.DataFrame(total_portfolio)
            self.total_mt5_portfolio["time_msc"] = pd.to_datetime(
                self.total_mt5_portfolio["time_msc"],
            )

    def format_mt5_portfolio(self):
        """Formats mt5 portfolio."""
        tp = self.total_mt5_portfolio

        # TODO refactor better for hedge
        if self.total_mt5_portfolio is None:
            self.gym_portfolio[:, 5:] = self.gym_portfolio[:, 5:] * 0.0
            self.gym_portfolio_hedge[:, 5:] = self.gym_portfolio_hedge[:, 5:] * 0.0
        else:
            portfolio = tp[tp["comment"] == self.agent_version].copy()
            if len(portfolio) > 0:
                self.gym_portfolio = self.update_gym_portfolio(
                    self.gym_portfolio,
                    portfolio,
                )

            else:
                self.gym_portfolio[:, 5:] = self.gym_portfolio[:, 5:] * 0.0

            portfolio = tp[tp["comment"] == f"{self.agent_version}_hedge"].copy()
            if len(portfolio) > 0:
                self.gym_portfolio_hedge = self.update_gym_portfolio(
                    self.gym_portfolio_hedge,
                    portfolio,
                )
            else:
                self.gym_portfolio_hedge[:, 5:] = self.gym_portfolio_hedge[:, 5:] * 0.0
