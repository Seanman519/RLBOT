"""MT5 Connector.

Interacts with MT5 terminal including:
- opening and closing positions and orders
- extracting data

Note:
- times are not in this class because different brokers have different time offsets
compared to utc.

"""
from __future__ import annotations

from datetime import datetime
from time import sleep

import MetaTrader5 as mt5
import pandas as pd
import pytz


class MT5Connector:
    """MT5 Handler."""

    def __init__(self, mt5_config):
        """Init.

        Args:
            config

        Returns:
            None

        """
        for k, v in mt5_config.items():
            setattr(self, k, v)
        # self.hour_diff = mt5_hour_diff(broker=self.broker)
        self.positions = None

    def initialize(self):
        """Initialize and log into MT5."""
        mt5_init = {}
        mt5_init["path"] = self.path
        mt5_init["login"] = self.login
        mt5_init["server"] = self.server
        mt5_init["password"] = self.password
        mt5_init["portable"] = True

        if not mt5.initialize(**mt5_init, timeout=10000):
            mt5.shutdown()
            return False, mt5.last_error()
        else:
            return True, None

    def check_mt5(self):
        """Check if MT5 needs to be reset."""
        t_info = mt5.terminal_info()
        if t_info is None:
            return False, self.initialize()
        else:
            return True, t_info

    def open_position(
        self,
        action,
        symbol,
        vol,
        agent_version,
        port_ind,
        deviation,
        tp_points=None,
        sl_points=None,
    ):
        """Open position.

        https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py

        Args:
            action (str)
                'buy' or 'sell'
            symbol (str)
            mult (float)
                how many multiples of the trade lot to trade
            comment (str)
                must be a integer string - used to help the algorithm determin what
                order to arrange positions to make it same as the gym env

        Returns:
            dict
                result_dict response from MT5

        """
        # prepare the buy request structure
        info = mt5.symbol_info(symbol)

        if action == "buy":
            trade_type = mt5.ORDER_TYPE_BUY
            price = info.ask
            mult = 1
        elif action == "sell":
            trade_type = mt5.ORDER_TYPE_SELL
            price = info.bid
            mult = -1

        # point = info.point
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": vol,
            "type": trade_type,
            "price": price,
            "deviation": deviation,
            "comment": agent_version,
            "magic": port_ind,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        if sl_points is not None:
            request["sl"] = price - sl_points * info.point * mult

        if tp_points is not None:
            request["tp"] = price + tp_points * info.point * mult

        # send a trading request
        result = mt5.order_send(request)

        return result

    def open_position_with_retry(
        self,
        retry_num,
        action,
        symbol,
        vol,
        agent_version,
        port_ind,
        deviation,
        tp_points=None,
        sl_points=None,
    ):
        """Open position with retry.

        Args:
            action (str)
                'buy' or 'sell'
            symbol (str)
            mult (float)
                how many multiples of the trade lot to trade
            comment (str)
                must be a integer string - used to help the algorithm determin what
                order to arrange positions to make it same as the gym env

        Returns:
            dict
                result_dict response from MT5

        """
        success = False
        max_tries = 0
        ret_comment = []
        while (max_tries < retry_num) & (success is False):
            result = self.open_position(
                action,
                symbol,
                vol,
                agent_version,
                port_ind,
                deviation,
                tp_points,
                sl_points,
            )
            if result.retcode == 10009:
                result = result._asdict()
                result["request"] = result["request"]._asdict()
                return result
            else:
                ret_comment.append(result.comment)
                max_tries += 1
                sleep(0.1)

        return {
            "return_comment": ret_comment,
            "retcode": 0,
        }

    def close_position(self, position_id, vol, deviation, port_ind, agent_version):
        """Close position.

        Args:
            positions (pd.Series)
            mult (float)
            comment (str)
                must be a integer string - used to help the algorithm determin what
                order to arrange positions to make it same as the gym env

        Returns:
            dict
                result_dict response from MT5

        """
        if self.positions is None:
            _ = self.get_positions()

        pos = self.positions[self.positions["identifier"] == position_id].iloc[0]
        symbol = pos["symbol"]
        info = mt5.symbol_info(symbol)
        action = pos["type"]

        if action == 0:
            # buy
            trade_type = mt5.ORDER_TYPE_SELL
            price = info.bid

        elif action == 1:
            # sell
            trade_type = mt5.ORDER_TYPE_BUY
            price = info.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": vol,
            "type": trade_type,
            "position": int(position_id),
            "price": price,
            "deviation": deviation,
            "magic": port_ind,
            "comment": agent_version,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        # send a trading request
        result = mt5.order_send(request)
        return result

    def close_position_with_retry(
        self,
        retry_num,
        position_id,
        vol,
        deviation,
        port_ind,
        agent_version,
    ):
        """Close position with retry.

        Args:
            positions (pd.Series)
            mult (float)
            comment (str)
                must be a integer string - used to help the algorithm determin what
                order to arrange positions to make it same as the gym env

        Returns:
            dict
                result_dict response from MT5

        """
        success = False
        max_tries = 0
        ret_comment = []
        while (max_tries < retry_num) & (success is False):
            result = self.close_position(
                position_id,
                vol,
                deviation,
                port_ind,
                agent_version,
            )
            if result.retcode == 10009:
                result = result._asdict()
                result["request"] = result["request"]._asdict()
                return result
            else:
                ret_comment.append(result.comment)
                max_tries += 1
                sleep(0.1)
        return {
            "return_comment": ret_comment,
            "retcode": 0,
        }

    def get_positions(self):
        """Get portfolio.

        Since we never change position, we don't need
        position_time_update (because that is same as position_time)
        position magic is for the index according to portfolio array
        comment is for agent
        since we don't allow for swaps, see identifier
        https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties
        #enum_position_reason

        Args:
            None

        Returns:
            pd.Dataframe
                all positions
        """
        positions = mt5.positions_get(group="**")
        if len(positions) > 0:
            positions = pd.DataFrame(
                list(positions),
                columns=positions[0]._asdict().keys(),
            )
            positions.sort_values(by=["comment", "magic"], inplace=True)
            positions["time_msc"] = pd.to_datetime(positions["time_msc"], unit="ms")
            positions.drop(
                ["time_update_msc", "time_update", "time", "reason", "ticket"],
                axis=1,
                inplace=True,
            )
            self.positions = positions
            return positions
        else:
            self.positions = None
            return pd.DataFrame()

    def get_tick_data(self, symbol, dt0, dt1):
        """Get tick data.

        #TODO probably more efficient to use dt0 and dt1 as ms from epoch rather
        than string, but will be harder to read

        #TODO investigate whether different brokers require different timezone
        encodings

        Args:
            symbol (str)
            dt0 (str)
                timezone must be UTC
            dt1 (str)
                timezone must be UTC

        Returns:
            np.array
                tick_data

        """
        dt0 = datetime.strptime(dt0, "%Y-%m-%d %H:%M:%S.%f")
        dt0 = pytz.utc.localize(dt0)

        dt1 = datetime.strptime(dt1, "%Y-%m-%d %H:%M:%S.%f")
        dt1 = pytz.utc.localize(dt1)

        counter = 0
        while counter < 10:
            try:
                df = mt5.copy_ticks_range(symbol, dt0, dt1, mt5.COPY_TICKS_ALL)
                if len(df) > 0:
                    df = pd.DataFrame(df).drop("time", axis=1)
                    return df
            except Exception:
                counter += 1
                sleep(10)
