"""Function relating to aerospike."""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from rlbot.utils.logging import get_logger

logger = get_logger(__name__)


def get_records_in_aerospike(config, client):
    """Get records in aerospike.

    Counts number of rows in aerospike table

    #TODO make it work if multiple version exist in database

    Args:
        data_config (dict)
        client (obj)
            aerospike client

    Returns:
        int

    """
    # save to static file because the windows / wine process cannot access Aerospike
    max_data_ind = client.info("sets")
    # get the record for the first namespace
    max_data_ind = max_data_ind[list(max_data_ind.keys())[0]][1]
    # filter to the correct set when there are multiple sets
    max_data_ind = max_data_ind.split("set=")
    # filter out short keys
    max_data_ind = [x for x in max_data_ind if len(x) > 6]
    # filter to correct version
    max_data_ind = [x for x in max_data_ind if x[:7] == f"{config.agent_version}:"]
    # if running for first time, i.e. building data, array will be empty
    if len(max_data_ind) > 0:
        max_data_ind = max_data_ind[0]
        # get number of objects
        max_data_ind = int(max_data_ind.split(":")[1].split("=")[-1]) - 1
        return max_data_ind
    else:
        return 0


def search_aerospike_for_dt(config, client, dt, start_val=None):
    """Search aerospike for date.

    When uploading data to a table that already exists, find the index that
    corresponds to the specified datetime, dt

    Args:
        config (pydantic.BaseModels):
            as defined in 'agent_config.py'
        client (aerospike.Client):
            client for downloading and inserting records
        dt (pd.DateTime):
            each record is read and compared to this datetime
        start_val (int):
            the table index at which the scanning start

    Returns:
        int
            table index for dt, the input datetime

    """
    max_data_ind = get_records_in_aerospike(config, client)
    if start_val is not None:
        max_data_ind = min(start_val, max_data_ind)
    if max_data_ind == 0:
        return 0

    ind_offset = None
    for i in tqdm(np.arange(max_data_ind, -1, -1)):
        key = (
            config.aerospike.namespace,
            config.aerospike.set_name,
            int(i),
        )
        try:
            _, _, bins = client.get(key)
        except Exception as e:
            logger.critical(str(repr(e)))
            bins = {}
        if "date" in bins:
            if dt == bins["date"]:
                ind_offset = i
                break

    return ind_offset
