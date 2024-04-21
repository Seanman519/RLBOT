"""Time.

Functions for manipulating time including:
- rounding
- timezones of different brokers
- waiting

"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta


def tz_diff(date, tz1, tz2):
    """Timezone time difference.

    Returns the difference in hours between timezone1 and timezone2
    for a given date.

    Args:
        date (np.datetime)
            date must be a datetime object
        tz1 (str)
        tz2 (str)

    Returns:
        float
            number of hours between the two timzones
    """
    return (tz1.localize(date) - tz2.localize(date).astimezone(tz1)).seconds / 3600


def mt5_hour_diff(broker="metaquotes"):
    """Time difference.

    Args:
        None

    Returns:
        int
            number of hours between UTC and broker clock

    """
    dt = datetime.now()
    utc = pytz.timezone("UTC")

    if broker == "metaquotes":
        eet = pytz.timezone("Europe/Helsinki")
        return int(tz_diff(dt, utc, eet))
    else:
        raise Exception(f"Set up time difference for {broker}")


def ceil_dt(dt, unit):
    """Round up the date by specified timeframe.

    Used to round data when extracting data

    Args:
        dt (np.datetime)
        unit (str)

    Returns:
        np.datetime

    """
    timezone = pytz.timezone("Etc/UTC")
    if unit == "day":
        dt = dt + relativedelta(days=1)
        return datetime(year=dt.year, month=dt.month, day=dt.day, tzinfo=timezone)
    elif unit == "month":
        dt = dt + relativedelta(months=1)
        return datetime(year=dt.year, month=dt.month, day=1, tzinfo=timezone)
    elif unit == "year":
        return datetime(year=dt.year + 1, month=1, day=1, tzinfo=timezone)
    elif unit == "decade":
        return datetime(
            year=dt.year + 10 - (dt.year % 10),
            month=1,
            day=1,
            tzinfo=timezone,
        )
    else:
        print("Bad date ceiling unit")


def wait_till_action_time(trade_timeframe, trade_time_offset, dt):
    """Get wait time.

    Calculate how many seconds to wait

    #TODO only works for trade frequencies - make more flexible
    # for strategies with action times > 1 min

    Args:
        trade_timeframe (str)
            polars format of time, must be in seconds. '10s' would indicate that an
            action can be taken every 10 seconds.
        trade_time_offset (str)
            polars format of time, must be in seconds. If the trade_timeframe = '10s'
            and trade_time_offset = '3s', then an action can be taken on the
            3rd, 13th, 23rd, 33rd, 43rd and 53rd seconds of each minute
        dt (pd.datetime or datetime.datetime)

    Returns
        bool
            True if the next time period is sleep,
            i.e. if weekend or market close or other no-trade
            period
        int
            number of seconds to sleep

    """
    # function returns whether to continue or not
    # skip if weekend
    if dt.weekday() >= 5:
        return True, 5

    trade_timeframe = float(trade_timeframe[:-1])
    trade_time_offset = float(trade_time_offset[:-1])
    # sleep until trade time
    seconds = (dt.second + dt.microsecond * 1e-6) % trade_timeframe
    if seconds > trade_time_offset:
        sleep_t0 = trade_timeframe + trade_time_offset
    else:
        sleep_t0 = trade_time_offset

    return False, np.clip(sleep_t0 - seconds, 0, None)


def get_current_mt5_time(hour_diff, format="datetime"):
    """Get current mt5 time.

    Args:
        format (str)
            'datetime' or 'str' for format

    Returns
        pd.datetime or str

    """
    t = pd.to_datetime("now", utc=True) + pd.Timedelta(hours=hour_diff)
    if format == "str":
        t = t.strftime("%Y-%m-%d %H:%M:%S.%f")
    return t


def ceil_timestamp(t, timeframe, trade_time_offset=3):
    """Round timestamp to the next timeframe.

    For example if the timestamp is 2023-01-01 10:20:25.123 and the feature_timeframe =
    10S, then the timestamp is rounded up to 2023-01-01 10:20:33.
    # TODO investigate whether I should be zeroing the microseconds - feels wrong
    Args:
        t (pd.Timestamp):
            tz unaware timestamp
        timeframe (str):
            string corresponding to pandas' timeframes https://pandas.pydata.org/
            pandas-docs/version/1.5/user_guide/timeseries.html#timeseries-offset-aliases

    Returns:
        pd.Timestamp

    """
    # t = t.replace(microsecond=0)-pd.Timedelta(seconds=trade_time_offset)
    t -= pd.Timedelta(seconds=trade_time_offset)
    timeframe = timeframe.replace("m", "T")
    t = t.ceil(timeframe) + pd.Timedelta(seconds=trade_time_offset)
    return t
