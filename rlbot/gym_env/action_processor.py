"""Action processor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


def build_pos_arrs(trade_config):
    """Build position array.

    The array is the gym environment representation of the portfolio.
    Each row is a position that can be traded.
    The columns should contain all the columns to keep track of active positions.

    Args
        trade_config (dict)

    Returns
        np.array
            the number of rows corresponds to number of tradeable tickers. columns:
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val

    """
    portfolio = np.zeros((len(trade_config.portfolio), 13), dtype="float64")
    for i in range(len(trade_config.portfolio)):
        tc = trade_config.portfolio[i]
        portfolio[i, 0] = i
        portfolio[i, 1] = tc.symbol_index
        portfolio[i, 2] = tc.pip_val
        portfolio[i, 3] = tc.max_short
        portfolio[i, 4] = tc.max_long
    return portfolio


def build_action_map(trade_config):
    """Build action map.

    Action map translates the agent prediction (i.e. int for discrete action) into
    a gym action (i.e. hold, open, close, etc.). Each row is the agent prediction,
    i.e. will have same length as the discrete action space. Each column describes
    the gym action, see below.

    Args
        trade_config (dict)

    Returns
        np.array
            Each row corresponds to a different action. Hold position is always index 0.
            The 4 columns from left to right:
            - portfolio_index
            - position direction: -1 for short and 1 for long
            - open or close: 'O' for open, 'C' for close
            - position magnitude: 1 or higher for lot size of each

    """
    # place holder for hold, index 0 is always hold
    action_map = [[-1, 0, 0, 0]]
    for j in range(len(trade_config.portfolio)):
        tc = trade_config.portfolio[j]
        for i in range(abs(tc.max_short)):
            # index , pos_dir: short (-1) or long (1),
            # order_type: open (1) or close (-1), pos_size
            action_map = action_map + [[j, -1, 1, i + 1], [j, -1, -1, i + 1]]
        for i in range(tc.max_long):
            action_map = action_map + [[j, 1, 1, i + 1], [j, 1, -1, i + 1]]

    return np.array(action_map, dtype="int")


@njit(nogil=True, cache=True, fastmath=True)
def update_portfolio_time_and_price(portfolio, curr_price, time_int):
    """Update portfolio time and price.

    At each timestep, update the portfolio based on the current price and time, i.e.
    the current value of positions and the holding time of open positions.

    Args
        portfolio (np.array):
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val
        current_price (np.array):
            array of value of all tickers
        time_int (int):
            current time in seconds

    Returns
        np.array
            same portfolio with updated position value and time held

    """
    # update portfolio time and price
    for port_ind in range(len(portfolio)):
        symbol_ind = int(portfolio[port_ind, 1])
        pos_dir = portfolio[port_ind, 6]

        if pos_dir != 0:
            # if long, then close using bid price
            if pos_dir == 1:
                price = curr_price[symbol_ind, 0]
            # if short, then close using ask price
            elif pos_dir == -1:
                price = curr_price[symbol_ind, 1]

            # set current price
            portfolio[port_ind, 11] = price

            # set average current value in pips
            portfolio[port_ind, 12] = price - portfolio[port_ind, 10]
            portfolio[port_ind, 12] /= portfolio[port_ind, 2] * pos_dir

            # set curr time
            portfolio[port_ind, 8] = time_int
            # set hold time
            portfolio[port_ind, 9] = (time_int - portfolio[port_ind, 7]) / 10

    return portfolio


@njit(nogil=True, cache=True, fastmath=True)
def exec_action(action_map, portfolio, action, curr_price, time_int, comm):
    """Execute gym action.

    Modify the portfolio based on the agent prediction. The steps are:
        - input action (i.e. an int between 0 and the discrete action space)
        - map the action to the action_map to translate it to the gym portfolio
        change, i.e. open or close a position within portfolio
        - modify and return the portfolio

    Args
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val
        action (int)
            action which corresponds to index of action_map
        curr_price (np.array)
            array of ticker prices
        time_int (int)
            current time in seconds
        comm (float)

    Returns
        portfolio (np.array)
        reward (np.float)
        trade_journal_entry  (np.array)
            for tracking closed positions

    """
    port_ind, pos_dir, order_type, pos_size = action_map[action]

    # get the bid ask price of the relevant ticker
    symbol_ind = int(portfolio[port_ind, 1])

    reward = 0
    trade_journal_entry = None

    # If hold, only update portfolio value
    if action == 0:
        portfolio = update_portfolio_time_and_price(portfolio, curr_price, time_int)

    # open a position
    if order_type == 1:
        # if long, then open using ask price
        if pos_dir == 1:
            price = curr_price[symbol_ind, 1]
        # if short, then open using bid price
        elif pos_dir == -1:
            price = curr_price[symbol_ind, 0]

        # set open price
        portfolio[port_ind, 10] = price

        # set open_time
        portfolio[port_ind, 7] = time_int
        # print("exec action open time:", portfolio[port_ind,7])

        # set position size
        # TODO assert that adding a position doesn't go over max position
        portfolio[port_ind, 5] = pos_size

        # set position direction
        portfolio[port_ind, 6] = pos_dir

        # update portfolio time and values which returns portfolio
        portfolio = update_portfolio_time_and_price(portfolio, curr_price, time_int)

    # partial or full close a position
    elif order_type == -1:
        # update portfolio time and values which returns portfolio
        portfolio = update_portfolio_time_and_price(portfolio, curr_price, time_int)

        # reward = current_price * pos_size
        reward = (portfolio[port_ind, 12] - comm) * pos_size

        # trade journal for tracking closed positions
        trade_journal_entry = portfolio[port_ind].copy()
        trade_journal_entry[5] = pos_size

        # reduce position size
        portfolio[port_ind, 5] = max(portfolio[port_ind, 5] - pos_size, 0)

        # if there are multiple positions, keep certain values,
        # otherwise, reset to 0
        if portfolio[port_ind, 5] == 0:
            # reset position direction
            portfolio[port_ind, 6:] = 0

    return portfolio, reward, trade_journal_entry


def format_portfolio(symbol_info, portfolio):
    """Format portfolio.

    Convert np.array to pd.Dataframe

    Args
        data_config (dict)
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val

    Returns
        pd.DataFrame

    """
    df = pd.DataFrame(
        portfolio,
        columns=[
            "port_ind",
            "symbol",
            "pip_val",
            "max_short",
            "max_long",
            "pos_size",
            "pos_dir",
            "open_time",
            "curr_time",
            "hold_time",
            "open_price",
            "curr_price",
            "curr_val",
        ],
    )
    int_cols = [
        "port_ind",
        "symbol",
        "max_short",
        "max_long",
        "pos_size",
        "pos_dir",
        "open_time",
        "curr_time",
        "hold_time",
    ]
    df[int_cols] = df[int_cols].astype(int)
    symbol_info = [x.symbol for x in symbol_info]
    df["symbol"] = [symbol_info[int(x)] for x in df["symbol"].tolist()]
    for col in ["open_time", "curr_time"]:
        df[col] = df[col].replace(0.0, np.nan)
        df[col] = pd.to_datetime(df[col], unit="s")

    return df


def make_action_labels(config, action_map, portfolio):
    """Make action labels.

    Converts action_map to human readable text

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val

    Returns
        List

    """
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    action_labels = []
    for a_ind in range(len(action_map)):
        if a_ind == 0:
            action_labels.append("Hold")
        else:
            a = action_map[a_ind]
            action_str = ""
            # portfolio index
            p_ind = int(a[0])

            symbol = [x.symbol for x in config.symbol_info][int(portfolio[p_ind, 1])]
            action_str += symbol

            # pos_dir
            action_str += "_Short" if a[1] == -1 else "_Long"

            # order type
            action_str += "_Close" if a[2] == -1 else "_Open"

            # pos_size
            action_str += "_" + str(a[3])

            # add portfolio index at the end because loki doesnt like digits
            # as first character
            action_str += "_" + str(a[0])

            action_labels.append(action_str)
    return action_labels
