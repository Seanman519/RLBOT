from __future__ import annotations

import os
from pathlib import Path

from releat.utils.time import mt5_hour_diff

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
}

broker_timezones = {
    "metaquotes": mt5_hour_diff(broker="metaquotes"),
}

mt5_creds = {
    "metaquotes": {
        "demo": {
            "path": f"{root_dir}/data/platforms/mt5/0/MetaTrader 5/terminal64.exe",
            "server": "MetaQuotes-Demo",
            "login": 5023954431,
            "password": "7kRhKh+g",
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
}
