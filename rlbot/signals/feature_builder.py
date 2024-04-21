"""Feature Builder.

Class that transforms tick data into features

"""
from __future__ import annotations

import _pickle as cPickle
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from rlbot.data.inference import init_feature_dict
from rlbot.data.inference import make_feature_obs
from rlbot.data.inference import update_feature_dict
from rlbot.gym_env.action_processor import build_action_map
from rlbot.gym_env.action_processor import build_pos_arrs
from rlbot.gym_env.mask import make_mask
from rlbot.gym_env.obs_processor import portfolio_to_model_input
from rlbot.utils.configs.config_builder import load_config
from rlbot.utils.logging import get_logger

logger = get_logger(__name__, log_level=logging.INFO)


class FeatureBuilder:
    """Feature Builder.

    Transforms tick data into gym observations at inference time.

    It also allows for hedging of the same strategy, i.e. trade in smaller lot sizes
    in the opposite position to your current position

    """

    def __init__(self, agent_version):
        """Init.

        Args:
            self:
                None
            agent_version (str):
                agent version, i.e. 't00001'. Configuration is loaded from this
                directory from the agents folder

        Returns:
            None

        """
        # tick data extracted from broker
        self.tick_data = None

        # feature data in intervals of the trade timeframe
        self.feature_data = None

        # feature data in intervals of the interval timeframe
        self.gym_data = None

        # Pydantic object with all configs
        self.config = load_config(agent_version, enrich_feat_spec=True, is_training=False)

        # array to map the output from the rl agent (integer) to a trade action
        # (i.e. open or close a long or short position)
        self.action_map = build_action_map(self.config.trader)

        # gym representation of the open positions
        self.gym_portfolio = build_pos_arrs(self.config.trader)

        # if a hedge position is also being traded, keep track of positions separately
        # here
        self.gym_portfolio_hedge = build_pos_arrs(self.config.trader)

        # if the hold time of a position is less than a minute, set to True
        self.must_hold = False

        # if close to session close time or weekend or other criteria, force close
        # active positions, one at a time
        self.must_close = False

    def partial_update_feature_data(self):
        """Partially updates feature.

        This is separated because its used in the initial and subsequent update
        feature functions. This only updates features that are build from tick data.
        Feature is updated on trade timframe at a time

        """
        # last timestampe from the existing feature data dict
        last_feature_t = pd.to_datetime(self.feature_data[0][0]["time_msc"][-1])

        # if the current time > time of the last feature
        # continually update the feature_data dict until the feature time is
        # larger than now
        while self.now > last_feature_t:
            # if the feature extends over the weekend, then skip iterate over the hours
            next_feat_t = last_feature_t + pd.Timedelta(
                self.config.raw_data.trade_timeframe,
            )
            while next_feat_t.dayofweek >= 5:
                if next_feat_t.hour == 23:
                    next_feat_t += pd.Timedelta(self.config.raw_data.trade_timeframe)
                else:
                    next_feat_t += pd.Timedelta(hours=1)

            logger.info(f"updating tick feature: {self.now} > {next_feat_t}")
            # updates the last record of the feature data
            self.feature_data = update_feature_dict(
                self.config,
                self.tick_data,
                self.feature_data,
                next_feat_t,
            )
            # a bit hacky but used to capture sparse records at the session open with
            # no tick data
            if next_feat_t > last_feature_t:
                # prevent infinite loop if no tick data at session open
                last_feature_t = next_feat_t
            else:
                # updates last feature time
                last_feature_t = pd.to_datetime(self.feature_data[0][0]["time_msc"][-1])

    def init_feature_data(self):
        """Initialise data.

        Load a pre-built feature dict from local storage. This is useful for larger
        models where the features can take longer to generate. Otherwise build
        dict from scratch

        """
        f = f"{self.config.paths.trader_log_dir}/init_feature_data.cpkl"
        if os.path.exists(f):
            with open(f, "rb") as fobj:
                self.feature_data = cPickle.load(fobj)
        else:
            # if path file does not exist, then create feat dict and write to local
            # storage
            self.feature_data = init_feature_dict(self.config, self.tick_data, self.now)
            with open(f, "wb") as fobj:
                cPickle.dump(self.feature_data, fobj)

        # update tick based feature data
        _ = self.partial_update_feature_data()

        # reset gym representation of positions for hedged and unhedged positions
        _ = self.reset_gym_pos_val(hedge=True)
        _ = self.reset_gym_pos_val(hedge=False)

        # save file locally, so its faster to restart
        with open(
            f"{self.config.paths.trader_log_dir}/init_feature_data.cpkl",
            "wb",
        ) as fobj:
            cPickle.dump(self.feature_data, fobj)

    def update_feature_data(self):
        """Update data."""
        _ = self.partial_update_feature_data()

        # update pos vals
        pos_val = portfolio_to_model_input(self.gym_portfolio).tolist()
        self.raw_pos_vals.append(pos_val)
        self.raw_pos_vals = self.raw_pos_vals[-20:]

        # hedge
        pos_val_hedge = portfolio_to_model_input(self.gym_portfolio_hedge).tolist()
        self.raw_pos_vals_hedge.append(pos_val_hedge)
        self.raw_pos_vals_hedge = self.raw_pos_vals_hedge[-20:]

        with open(
            f"{self.config.paths.trader_log_dir}/init_feature_data.cpkl",
            "wb",
        ) as fobj:
            cPickle.dump(self.feature_data, fobj)

    def make_gym_mask(self, portfolio):
        """Make gym mask.

        Args:
            portfolio (np.array):
                gym represetnation of the positions

        Returns:
            np.array:
                binary mask where 1 = action available and 0 = action not allowed

        """
        a = np.any(
            (portfolio[:, 9] <= self.config.gym_env.min_hold_t) & (portfolio[:, 6] != 0),
        )
        must_hold = self.must_hold | np.bool(a)
        # TODO make more parametric
        a = portfolio[:, 9].max() > self.config.gym_env.max_hold_t
        must_close = self.must_close | np.bool(a)
        return make_mask(
            self.action_map,
            portfolio,
            self.config.gym_env.stop_val,
            must_hold,
            must_close,
            False,  # pos_size_filter
            0,  # pos_dir filter
        )

    def make_gym_data(self):
        """Get gym data.

        Reformats the data into the gym observation representation.

        """
        gym_obs_data = make_feature_obs(self.config, self.feature_data)

        dt = self.now.strftime("%Y-%m-%d %H:%M:%S")
        date_arr = np.array(
            [
                np.round(float(dt[11:13]) / 23, decimals=6),
                np.round(float(dt[14:16]) / 59, decimals=6),
                np.round(float(dt[17:19]) / 59, decimals=6),
            ],
            dtype="float32",
        )
        gym_obs_data["date_arr"] = date_arr
        gym_obs_data["pos_val"] = self.raw_pos_vals
        gym_obs_data["pos_val_hedge"] = self.raw_pos_vals_hedge
        # TODO make parametric
        gym_obs_data["pos_val_no_pos"] = [[0.0, 0.0]] * 20
        gym_obs_data["mask"] = self.make_gym_mask(self.gym_portfolio)
        gym_obs_data["mask_hedge"] = self.make_gym_mask(self.gym_portfolio_hedge)
        # TODO improve how this is fetched
        portfolio = deepcopy(self.gym_portfolio)
        portfolio[:, 5:] = portfolio[:, 5:] * 0.0
        gym_obs_data["mask_no_pos"] = make_mask(
            self.action_map,
            portfolio,
            self.config.gym_env.stop_val,
            False,
            False,
            False,
            0,
        )

        return gym_obs_data

    def reset_gym_pos_val(self, hedge):
        """Reset gym pos val.

        Reset gym observation of gym value.

        """
        # TODO make parametric
        if hedge:
            self.raw_pos_vals = [[0.0, 0.0]] * 20
        else:
            self.raw_pos_vals_hedge = [[0.0, 0.0]] * 20
