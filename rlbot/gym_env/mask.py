"""Action masks."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(nogil=True, cache=True, fastmath=True)
def make_base_mask(action_map, portfolio, must_hold):
    """Make base mask.

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val
        must_hold (bool)
            whether force hold position

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    """
    # Note never have must_close = True and must_hold = True together
    # must_close = False

    action_mask = np.zeros((len(action_map),), dtype="float32")
    # hold is always available
    action_mask[0] = 1

    for port_ind in range(len(portfolio)):
        # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
        # open_time, curr_time, hold_time, open_price, curr_price, current_val
        pos_size = portfolio[port_ind, 5]

        # index , short (-1) or long (1),
        # open (1) or close (-1), pos_size
        a = action_map[:, 0] == port_ind

        # open
        if not must_hold:
            # filter to open actions only
            b = a & (action_map[:, 2] == 1)

            # if a position is already open, we can only open one in the same direction
            # For now we disable this because the current framework doesn't allow us to
            # hold multiple open_times, which makes tracking reward difficult
            # if pos_size > 0:
            #     b = b & (action_map[:,1]==portfolio[ind,6])
            # if the current position size is 0, then we can open position
            if pos_size == 0:
                # open for SHORT positions only
                c = action_map[:, 1] == -1
                # next pos size < max pos size
                c = c & ((pos_size + action_map[:, 3]) <= abs(portfolio[port_ind, 3]))
                c = b & c
                action_mask[c] = 1

                # open for LONG positions only
                c = action_map[:, 1] == 1
                # next pos size < max pos size
                c = c & ((pos_size + action_map[:, 3]) <= abs(portfolio[port_ind, 4]))
                c = b & c
                action_mask[c] = 1

        # close
        if not must_hold:
            # close must only apply to same position direction
            c = action_map[:, 1] == portfolio[port_ind, 6]
            # if position size > 0
            b = c & a & (pos_size > 0) & (pos_size >= action_map[:, 3])
            # filter to close actions only
            b = b & (action_map[:, 2] == -1)
            action_mask[b] = 1

    return action_mask


@njit(nogil=True, cache=True, fastmath=True)
def make_stop_loss_mask(action_map, portfolio, stop_loss):
    """Make stop loss mask.

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val
        stop_loss (float)
            if position value is smaller than stop loss, then close position

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    """
    # stop_loss = -10.0
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    # get the bid ask price of the relevant ticker
    action_mask = np.zeros((len(action_map),), dtype="float32")

    port_ind = np.argwhere(portfolio[:, 12] < stop_loss)[0][0]

    # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
    # open_time, curr_time, hold_time, open_price, curr_price, current_val
    # filter to relevant port ind
    a = action_map[:, 0] == port_ind
    # filter to close actions only
    a = a & (action_map[:, 2] == -1)
    # filter to positions in the same direction as the current position only
    a = a & (action_map[:, 1] == portfolio[port_ind, 6])
    # can close either 1 or more positions
    pos_size = portfolio[port_ind, 5]
    a = a & (pos_size > 0) & (pos_size >= action_map[:, 3])

    action_mask[a] = 1
    return action_mask


@njit(nogil=True, cache=True, fastmath=True)
def make_pos_size_mask(action_map):
    """Make pos size mask.

    Force the agent to only trade 1 lot (rather than 2 or 3)

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    # TODO rather than just lock for >1 I should just randomly sample it for
    one episode, may need another parameter for that

    """
    # stop_loss = -10.0
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    # get the bid ask price of the relevant ticker
    action_mask = np.ones((len(action_map),), dtype="float32")

    # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
    # open_time, curr_time, hold_time, open_price, curr_price, current_val
    # filter to pos size > 1
    a = action_map[:, 3] > 1
    # filter to open actions only
    a = a & (action_map[:, 2] == 1)

    action_mask[a] = 0
    return action_mask


@njit(nogil=True, cache=True, fastmath=True)
def make_pos_long_mask(action_map):
    """Make pos size mask.

    Force the agent to only trade 1 lot (rather than 2 or 3)

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    # TODO rather than just lock for >1 I should just randomly sample it for
    one episode, may need another parameter for that

    """
    # stop_loss = -10.0
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    # get the bid ask price of the relevant ticker
    action_mask = np.ones((len(action_map),), dtype="float32")

    # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
    # open_time, curr_time, hold_time, open_price, curr_price, current_val
    # filter to pos size > 1
    a = action_map[:, 1] == -1
    # filter to open actions only
    a = a & (action_map[:, 2] == 1)

    action_mask[a] = 0
    return action_mask


@njit(nogil=True, cache=True, fastmath=True)
def make_pos_short_mask(action_map):
    """Make pos size mask.

    Force the agent to only trade 1 lot (rather than 2 or 3)

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    # TODO rather than just lock for >1 I should just randomly sample it for
    one episode, may need another parameter for that

    """
    # stop_loss = -10.0
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    # get the bid ask price of the relevant ticker
    action_mask = np.ones((len(action_map),), dtype="float32")

    # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
    # open_time, curr_time, hold_time, open_price, curr_price, current_val
    # filter to pos size > 1
    a = action_map[:, 1] == 1
    # filter to open actions only
    a = a & (action_map[:, 2] == 1)

    action_mask[a] = 0
    return action_mask


@njit(nogil=True, cache=True, fastmath=True)
def make_episode_end_mask(action_map, portfolio):
    """Make episode end mask.

    If the agent is too close to episode end, then do not open any new positions and
    close existing positions

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    """
    # index , pos_dir: short (-1) or long (1),
    # order_type: open (1) or close (-1), pos_size
    # get the bid ask price of the relevant ticker
    action_mask = np.zeros((len(action_map),), dtype="float32")

    for port_ind in range(len(portfolio)):
        # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
        # open_time, curr_time, hold_time, open_price, curr_price, current_val
        # filter to relevant port ind
        a = action_map[:, 0] == port_ind
        # filter to close actions only
        a = a & (action_map[:, 2] == -1)
        # filter to positions in the same direction as the current position only
        a = a & (action_map[:, 1] == portfolio[port_ind, 6])
        # can close either 1 or more positions
        pos_size = portfolio[port_ind, 5]
        a = a & (pos_size > 0) & (pos_size >= action_map[:, 3])

        action_mask[a] = 1

    if sum(action_mask) == 0.0:
        action_mask[0] = 1
    return action_mask


@njit(cache=True,nogil=True, fastmath=True)
def np_any_axis0(x):
    """Numba compatible version of np.any(x, axis=0).

    https://stackoverflow.com/questions/61304720/workaround-for-numpy-
    np-all-axis-argument-compatibility-with-numba

    """
    out = np.zeros(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_or(out, x[i, :])
    return out


@njit(cache=True,nogil=True, fastmath=True)
def assess_must_actions(
    portfolio,
    ep_time,
    max_ep_step,
    min_hold_time,
    max_hold_time,
    curr_data_ind,
    max_data_ind,
):
    """Return must close and must hold.

    #TODO add in must close for end of day to avoid swap fee

    Positions must be closed if:
        - close to end of episode
        - longer than max hold time (arbitrary value defined by your strategy)
        - if reaching end of dataset (only relevant to algo evaluation)

    Positions must by held if:
        - opened less than the min hold time (prevents the algo from HFT)

    """
    # must close if about close to episode end or last datapoint
    must_close = False
    must_close = must_close | (ep_time >= max_ep_step - 50)
    must_close = must_close | np.any(portfolio[:, 9] >= max_hold_time)
    # if self.is_training:
    must_close = must_close | (curr_data_ind > max_data_ind - 50)

    # must hold if just opened position
    # portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size, pos_dir
    # open_time, curr_time, hold_time, open_price, curr_price, current_val
    a = portfolio[:, 5] != 0
    if a.sum() > 0:
        must_hold = min(portfolio[a, 9]) <= min_hold_time
    else:
        must_hold = False
    return must_hold, must_close


@njit(nogil=True, cache=True, fastmath=True)
def make_mask(
    action_map,
    portfolio,
    stop_loss,
    must_hold,
    must_close,
    mask_pos_size,
    mask_pos_dir,
):
    """Make base mask.

    Args:
        action_map (np.array)
            columns: index , pos_dir: short (-1) or long (1),
            order_type: open (1) or close (-1), pos_size
        portfolio (np.array)
            portfolio_index, symbol_index, pip_val, max_short, max_long, pos_size,
            pos_dir, open_time, curr_time, hold_time, open_price, curr_price, current_val
        stop_loss (float)
            if position value is smaller than stop loss, then close position
        must_hold (bool)
            whether force hold position
        must_close (bool)
            whether force close a position

    Returns:
        action_mask (np.array)
            binary mask of whether an action is allowable. 1 = yes, 1 = no

    """
    if min(portfolio[:, 12]) < stop_loss:
        mask = make_stop_loss_mask(action_map, portfolio, stop_loss)
    elif must_close:  # episode end
        mask = make_episode_end_mask(action_map, portfolio)
    else:
        mask = make_base_mask(action_map, portfolio, must_hold)

    # Can only trade a maximum of X position size
    if mask_pos_size:
        mask = mask.astype(np.bool_) & make_pos_size_mask(action_map).astype(np.bool_)
        mask = mask.astype("float32")

    # ep can only long OR short
    if mask_pos_dir == -1:
        mask = mask.astype(np.bool_) & make_pos_short_mask(action_map).astype(np.bool_)
        mask = mask.astype("float32")
    elif mask_pos_dir == 1:
        mask = mask.astype(np.bool_) & make_pos_long_mask(action_map).astype(np.bool_)
        mask = mask.astype("float32")

    return mask
