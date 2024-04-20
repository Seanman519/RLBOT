"""MT5 api.

API so that data can be easily shared between linux and wine processes.

#TODO move to redis

"""
from __future__ import annotations

import _pickle as cPickle
import argparse

import pandas as pd
from flask import Flask
from flask import request
from flask import Response

from rlbot.connectors.mt5 import MT5Connector
from rlbot.utils.configs.constants import mt5_api_port_map

argParser = argparse.ArgumentParser(
    description="Start mt5 api",
)

argParser.add_argument("-b", "--broker", help="broker")
argParser.add_argument("-s", "--symbol", help="trading instrument")
args = argParser.parse_args()

app = Flask(__name__)


class MT5Api:
    """MT5 wrapper for flask api."""

    def __init__(self):
        """Initialize class."""
        self.conn = None

    def initialize(self, mt5_config):
        """Initialize MT5 connector."""
        self.conn = MT5Connector(mt5_config)
        status = self.conn.initialize()
        if status:
            return {"status": "ok"}, 200
        else:
            return {"status": "fail"}, 200

    def healthcheck(self):
        """Healthcheck."""
        status, tinfo = self.conn.check_mt5()
        if status:
            return tinfo._asdict()
        else:
            return str(tinfo)

    def open(self, data):
        """Open position."""
        return self.conn.open_position_with_retry(**data)

    def close(self, data):
        """Close position."""
        return self.conn.close_position_with_retry(**data)

    def get_positions(self):
        """Get positions."""
        df = self.conn.get_positions()
        if df is not None:
            return df.to_dict(orient="records")
        else:
            return {}

    def get_tick_data(self, data):
        """Get tick data."""
        df = self.conn.get_tick_data(**data)
        if df is None:
            df = {"error": "no tick data"}
        elif type(df) != pd.DataFrame:
            df = {
                "data": str(df),
                "error": "wrong data type",
            }
        elif len(df) == 0:
            df = {"error": "no tick data"}
        else:
            df = {"data": df}

        df = cPickle.dumps(df)

        return Response(df, content_type="application/octet-stream")


mt5c = MT5Api()


@app.route("/init", methods=["POST"])
def initialise_connector():
    """Initialize mt5 connector."""
    mt5_config = request.json
    return mt5c.initialize(mt5_config)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    """Checks health of mt5."""
    return mt5c.healthcheck()


@app.route("/open", methods=["POST"])
def open_position():
    """Open position."""
    data = request.json
    return mt5c.open(data)


@app.route("/close", methods=["POST"])
def close_position():
    """Close position."""
    data = request.json
    return mt5c.close(data)


@app.route("/get_positions", methods=["GET"])
def get_positions():
    """Get open positions."""
    return mt5c.get_positions()


@app.route("/get_tick_data", methods=["GET"])
def get_tick_data():
    """Get tick data."""
    data = request.json
    return mt5c.get_tick_data(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=mt5_api_port_map[args.broker][args.symbol], debug=False)
