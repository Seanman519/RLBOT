"""Feature config.

- 10S, 2T, 30T features for EURUSD
- small scale dataset for testing different types of features

"""
from __future__ import annotations

len_30s = 5
len_5m = 3

default_transforms = [
    {
        "name": "clip",
        "method": "percentile",
        "upper_lim": 99.9,
        "lower_lim": 0.1,
        "scale_factor": 1,
    },
    {
        "fillna": "zero",
        "name": "scale",
        "method": "PowerTransformer",
        "is_elementwise": True,
    },
    {
        "name": "scale",
        "method": "PiecewiseLinear",
    },
    {
        "name": "clip",
        "method": "value",
        "upper_lim": 3,
        "lower_lim": -3,
        "scale_factor": 0.5,
    },
]


feature_config = [
    {
        "timeframe": "30s",
        "stack_axis": 1,
        "simple_features": [
            {
                "name": "differencing",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["avg_price"],
                "output_shape": (len_30s, 1),
                "timeframe_mode": "rolling",
                "kwargs": {},
                "fillna": "forward",
                "transforms": default_transforms,
            },
            {
                "name": "one_hot",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["flags"],
                "output_shape": (len_30s, 3),
                "timeframe_mode": "rolling",
                "kwargs": {
                    "normalize": True,
                },
                "fillna": "zero",
                "transforms": default_transforms,
            },
            {
                "name": "mean",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["spread"],
                "output_shape": (len_30s, 1),
                "timeframe_mode": "rolling",
                "kwargs": {},
                "fillna": "forward",
                "transforms": default_transforms,
            },
        ],
    },
    {
        "timeframe": "5m",
        "stack_axis": 1,
        "simple_features": [
            {
                "name": "differencing",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["avg_price"],
                "output_shape": (len_5m, 1),
                "timeframe_mode": "rolling",
                "kwargs": {},
                "fillna": "forward",
                "is_train_data_pretransformed": False,
                "transforms": default_transforms,
            },
            {
                "name": "min",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["avg_price"],
                "output_shape": (len_5m, 1),
                "timeframe_mode": "rolling",
                "kwargs": {},
                "fillna": "zero",
                "transforms": default_transforms,
            },
            {
                "name": "max",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["avg_price"],
                "output_shape": (len_5m, 1),
                "timeframe_mode": "rolling",
                "kwargs": {},
                "fillna": "zero",
                "transforms": default_transforms,
            },
            {
                "name": "grad",
                "broker": "metaquotes",
                "symbol": "EURUSD",
                "inputs": ["avg_price"],
                "output_shape": (len_5m, 1),
                "timeframe_mode": "rolling",
                "kwargs": {
                    "min_num": 10,
                },
                "fillna": "zero",
                "transforms": default_transforms,
            },
        ],
    },
]
