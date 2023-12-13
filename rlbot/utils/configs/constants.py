"""Constants.

Static variables and parameters that are used throughout the app.

#TODO convert to yaml?

"""
from __future__ import annotations

import os
from pathlib import Path

from rlbot.utils.time import mt5_hour_diff

root_dir = str(Path(os.path.dirname(__file__)).parents[2])

s3_secret = None


mt5_api_port_map = {
    "metaquotes": {
        "general": 2000,
        "EURUSD": 2001,
        "AUDJPY": 2002,
        "USDJPY": 2003,
        "XAUUSD": 2004,
    },
    "ampglobal": {
        "general": 2100,
        "@CLE": 2101,
        "@DB": 2102,
        "@EP": 2103,
        "@ENQ": 2104,
    },
}

broker_timezones = {
    "metaquotes": mt5_hour_diff(broker="metaquotes"),
    "ampglobal": 0,
}

mt5_creds = {
    "metaquotes": {
        "demo": {
            "path": f"{root_dir}/data/platforms/mt5/0/MetaTrader 5/terminal64.exe",
            "server": "MetaQuotes-Demo",
            "login": 75342897,
            "password": "!p7nCzDf",
        },
    },
    "ampglobal": {
        "demo": {
            "path": f"{root_dir}/data/platforms/mt5/1/MetaTrader 5/terminal64.exe",
            "server": "AMPGlobalUSA-Demo",
            "login": 1432733,
            "password": "ndupedd4",
        },
    },
}

# Even though pip val may seem trivial for forex,
# It is different for futures contracts
trading_instruments = {
    "metaquotes": {
        "EURUSD": {
            "pip": 1e-4,
            "pip_val": 100,
            "contract_size": 100_000,
            "currency": "USD",
            "commission": 4.0,
            "maintenance": 100,
            "margin": 100,
        },
        "AUDJPY": {
            "pip": 1e-2,
            "pip_val": 10_000,
            "contract_size": 100_000,
            "currency": "JPY",
            "commission": 4.0,
            "maintenance": 100,
            "margin": 100,
        },
        "USDJPY": {
            "pip": 1e-2,
            "pip_val": 10_000,
            "contract_size": 100_000,
            "currency": "JPY",
            "commission": 4.0,
            "maintenance": 100,
            "margin": 100,
        },
        "XAUUSD": {
            "pip": 0.1,
            "pip_val": 100,
            "contract_size": 100,
            "currency": "USD",
            "commission": 4.0,
            "maintenance": 1000,
            "margin": 1000,
        },
    },
    "ampglobal": {
        "@CLE": {
            "pip": 0.1,
            "pip_val": 100,
            "contract_size": 1,
            "currency": "USD",
            "commission": 4.26,
            "maintenance": 7000,
            "margin": 1750,
        },
        "@DB": {
            "pip": 0.1,
            "pip_val": 100,
            "contract_size": 1,
            "currency": "USD",
            "commission": 1.68,
            "maintenance": 4450,
            "margin": 1112.5,
        },
        "@EP": {
            "pip": 0.1,
            "pip_val": 12.5,
            "contract_size": 1,
            "currency": "USD",
            "commission": 3.8,
            "maintenance": 15800,
            "margin": 1000,
        },
        "@ENQ": {
            "pip": 0.1,
            "pip_val": 50,
            "contract_size": 1,
            "currency": "USD",
            "commission": 3.8,
            "maintenance": 10600,
            "margin": 400,
        },
    },
}
