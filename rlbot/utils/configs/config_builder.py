"""Config Builder.

Combine all config python files into one object

"""
from __future__ import annotations

import importlib
import os
import sys
from functools import partial

import numpy as np
from gymnasium import spaces

from rlbot.data.transformers import enrich_all_feature_configs
from rlbot.gym_env.action_processor import build_action_map
from rlbot.utils.configs.constants import root_dir
from rlbot.utils.configs.constants import trading_instruments
from rlbot.utils.configs.data_models import AerospikeConfig
from rlbot.utils.configs.data_models import AgentConfig
from rlbot.utils.configs.data_models import FeatureGroupConfig
from rlbot.utils.configs.data_models import GymEnvConfig
from rlbot.utils.configs.data_models import MT5Config
from rlbot.utils.configs.data_models import Paths
from rlbot.utils.configs.data_models import PositionConfig
from rlbot.utils.configs.data_models import RawDataConfig
from rlbot.utils.configs.data_models import RedisConfig
from rlbot.utils.configs.data_models import SimpleFeatureConfig
from rlbot.utils.configs.data_models import SymbolSpec
from rlbot.utils.configs.data_models import TraderConfig
from rlbot.utils.configs.data_models import TransformerConfig


def make_box_space(min_obs_val, max_obs_val, shape):
    """Make box space for gym env.

    Args:
        min_obs_val (np.float32):
            minimum observed value (clip value + buffer to account for precision errors)
        max_obs_val (np.float32):
            maximum observed value
        shape (tuple):
            indicates the shape of the box space (could be inputs or actions)

    Returns:
        gymnasium.spaces.box

    """
    return spaces.Box(low=min_obs_val, high=max_obs_val, shape=shape, dtype=np.float32)


def get_feat_group_shape(feature_spec):
    """Get feature group shape.

    Args:
        feature_spec (List(dict)): list from the feature_config.py file

    Returns:
        dict: for each feature group index, the expected shape of the array
    """
    symbol_structure = {}
    for feat_ind in range(len(feature_spec)):
        fcs = feature_spec[feat_ind]
        timeframe = fcs.timeframe
        assert timeframe not in symbol_structure
        i0 = fcs.simple_features[0].output_shape[0]
        j0 = 0
        for ind in range(len(fcs.simple_features)):
            fc = fcs.simple_features[ind]
            i1, j1 = fc.output_shape
            assert i0 == i1
            j0 += j1
        symbol_structure[str(feat_ind)] = (i0, j0)
    return symbol_structure


def make_save_paths(agent_version):
    """Make save paths.

    Args:
        agent_version (str): agent version

    Returns:
        dict: subdirectories for raw, intermediate data, models, etc.
    """
    agent_root_dir = f"{root_dir}/data/agent/{agent_version}"

    paths = {
        # common file paths
        "root_dir": root_dir,
        # tick data
        "tick_data_dir": f"{root_dir}/data/tick_data",
        # experiment specific file paths
        # location for rl algo checkpoints
        "algo_dir": f"{agent_root_dir}/algo",
        # agent test data
        "eval_dir": f"{agent_root_dir}/eval",
        # trained features
        "feature_dir": f"{agent_root_dir}/features",
        # data for updating db
        "update_tick_data_dir": f"{agent_root_dir}/update_tick_data",
        # data for updating db
        "trader_log_dir": f"{agent_root_dir}/trade_log",
    }

    # create folders
    for _, path in paths.items():
        _ = os.makedirs(path, exist_ok=True)

    return Paths(**paths)


def get_ticker_info(feature_spec):
    """Get ticker info.

    Reads reference file with all the available tickers, times, lot sizes and other
    information

    For now we create a symbol_info_index as well because the gym environment uses numba
    which uses numerical indexes rather than test.
    #TODO streamline this in the future with a better design pattern.

    Args:
        feature_spec (List(dict)):
            config from the feature_config.py file.

    Returns:
        dict:
            symbols which are used in the model, along with relevant information, such
            as pip size, contract size, currency, etc. Also returns an index to make
            searching and referencing easier
    """
    agent_symbol_info = []
    agent_symbol_info_index = {}
    for i in range(len(feature_spec)):
        fcs = feature_spec[i]["simple_features"]
        for j in range(len(fcs)):
            symbol = fcs[j]["symbol"]
            broker = fcs[j]["broker"]
            a = symbol not in [x.symbol for x in agent_symbol_info]
            a = a & (broker not in [x.broker for x in agent_symbol_info])
            if a:
                symbol_info = trading_instruments[broker][symbol]
                symbol_info["symbol"] = symbol
                symbol_info["broker"] = broker
                agent_symbol_info_index[symbol] = len(agent_symbol_info)
                agent_symbol_info.append(SymbolSpec(**symbol_info))

    return {
        "symbol_info": agent_symbol_info,
        "symbol_info_index": agent_symbol_info_index,
    }


def make_agent_config(config, feature_spec):
    """Make agent config.

    Preprocessing and adding in some addition fields to the config before converting
    it into a class.

    Args:
        config (dict): from agent_config.py
        feature_spec (List(dict)): from feature_config.py

    Returns:
        obj:
            single config used through training and deployment process
    """
    config["paths"] = make_save_paths(config["agent_version"])
    config["aerospike"]["set_name"] = config["agent_version"]
    config["aerospike"] = AerospikeConfig(**config["aerospike"])
    config["redis"] = RedisConfig(**config["redis"])
    config["mt5"] = MT5Config(**config["mt5"])

    config = {**config, **get_ticker_info(feature_spec)}

    config["raw_data"] = RawDataConfig(**config["raw_data"])

    for i in range(len(feature_spec)):
        feat_group = feature_spec[i]
        feature_spec[i]["index"] = i
        timeframe = feat_group["timeframe"]
        for j in range(len(feat_group["simple_features"])):
            transforms = feature_spec[i]["simple_features"][j]["transforms"]
            transforms = [TransformerConfig(**x) for x in transforms]
            feature_spec[i]["simple_features"][j]["transforms"] = transforms
            feature_spec[i]["simple_features"][j]["timeframe"] = timeframe
            feature_spec[i]["simple_features"][j]["index"] = j
            feature_spec[i]["simple_features"][j] = SimpleFeatureConfig(
                **feature_spec[i]["simple_features"][j],
            )
        feature_spec[i] = FeatureGroupConfig(**feature_spec[i])

    config["features"] = feature_spec

    config["gym_env"] = GymEnvConfig(**config["gym_env"])

    # Trader Config
    trader_config = config["trader"]

    symbol_index_map = {}
    for i, val in enumerate(config["symbol_info"]):
        symbol_index_map[val.symbol] = [i, val.pip]

    for i in range(len(trader_config["portfolio"])):
        tc = trader_config["portfolio"][i]
        trader_config["portfolio"][i]["symbol_index"] = symbol_index_map[tc["symbol"]][0]
        trader_config["portfolio"][i]["pip_val"] = symbol_index_map[tc["symbol"]][1]
        trader_config["portfolio"][i] = PositionConfig(
            **trader_config["portfolio"][i],
        )
    config["trader"] = TraderConfig(**trader_config)

    # Make observation space
    symbol_structure = get_feat_group_shape(feature_spec)
    action_len = len(build_action_map(config["trader"]))
    p_make_box_space = partial(
        make_box_space,
        config["raw_data"].min_obs_val,
        config["raw_data"].max_obs_val,
    )
    obs_space = {
        # TODO make date representation parametric
        "date_arr": p_make_box_space((3,)),
        # TODO make the position value parametric
        "pos_val": p_make_box_space((1, 2)),
        "mask": p_make_box_space((action_len,)),
    }
    for k, v in symbol_structure.items():
        obs_space[k] = p_make_box_space(v)
    obs_space = spaces.Dict(obs_space)

    config["observation_space"] = obs_space
    config["action_space"] = spaces.Discrete(action_len)

    # populate rllib specific parameters
    config["rl_env"] = {
        "observation_space": obs_space,
        "action_space": spaces.Discrete(action_len),
    }
    config["rl_train"]["model"]["custom_model_config"][
        "symbol_structure"
    ] = symbol_structure
    # Input shape for agent model
    config["rl_train"]["model"]["custom_model_config"]["input_shape"] = {
        **symbol_structure,
        "date_arr": (3,),
        # TODO make the position value parametric + different views of portfolio
        "pos_val": (1, 2),
        "mask": (action_len,),
    }

    config = AgentConfig(**config)

    return config


def load_config(
    agent_version,
    enrich_feat_spec=False,
    is_training=True,
    load_model=False,
):
    """Load configs.

    Loads the config depending on whether the config is used for training or inference

    Args:
        agent_version (str):
            should be same as director
        enrich_feat_spec (bool):
            if True then add on the pre-calculated scaling and transform arrays
        is_training (bool):
            if True, use lots of resources to train model, else use fewer cpus, etc.

    Returns:
        obj:
            class object with all configs required by agent

    """
    file_path = f"{root_dir}/agents/{agent_version}"
    sys.path.insert(0, file_path)
    configs = []

    for name in ["agent_config", "feature_config"]:
        spec = importlib.util.spec_from_file_location(
            name,
            f"{file_path}/{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        configs.append(getattr(module, name))

    agent_config, feature_spec = configs

    if not is_training:
        agent_config["gym_env"]["is_training"] = False
        agent_config["gym_env"]["log_actions"] = True
        agent_config["rl_rollouts"] = {
            "num_rollout_workers": 0,
            "num_envs_per_worker": 1,
        }
        agent_config["rl_explore"] = {
            "explore": False,
        }
        agent_config["rl_resources"] = {
            # Resources
            "num_gpus": 0,
            "num_gpus_per_worker": 0,
            "num_cpus_per_worker": 1,
        }
        agent_config["rl_train"]["train_batch_size"] = 1

    config = make_agent_config(agent_config, feature_spec)

    if enrich_feat_spec:
        config = enrich_all_feature_configs(config)

    if load_model:
        # TODO fix the hacky import
        exec(f"from agents.{agent_version}.agent_model import AgentModel")
        return config, eval("AgentModel")
    else:
        return config
