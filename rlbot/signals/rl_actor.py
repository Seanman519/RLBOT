"""Predicts action."""
from __future__ import annotations

from glob import glob

import ray
from ray.rllib.algorithms.impala import ImpalaConfig as RLAlgorithmConfig
from ray.rllib.models import ModelCatalog

from rlbot.gym_env.action_processor import build_action_map
from rlbot.gym_env.gym_env import FxEnv
from rlbot.utils.configs.config_builder import load_config


class RLActor:
    """Predictor.

    Class for using the RL agent to predict an action.

    """

    def __init__(self, agent_version, address="local"):
        """Init.

        Args:
            agent_version (str):
                i.e. 't00001' - should match a folder that exists in the agent folder
            address (str):
                'local' or any other value to indicate whether to spin up ray locally
                (for testing) or to connect to an existing instance of ray.

        Returns:
            None

        """
        # start local ray instance (for testing) or connecting to an existn instance
        if address == "local":
            ray.init(local_mode=True, ignore_reinit_error=True)
        else:
            ray.init(address="auto", namespace="serve")

        # load config
        config, AgentModel = load_config(
            agent_version,
            enrich_feat_spec=True,
            is_training=False,
            load_model=True,
        )

        # register model
        ModelCatalog.register_custom_model("AgentModel", AgentModel)

        # reformat some configs
        env_config = {
            **config.rl_env,
            "env_config": dict(config),
        }

        # initialise RL agent
        self.trainer = (
            RLAlgorithmConfig()
            .training(**config.rl_train)
            .environment(env=FxEnv, **env_config)
            .framework(**config.rl_framework)
            .rollouts(**config.rl_rollouts)
            .exploration(**config.rl_explore)
            .reporting(**config.rl_reporting)
            .debugging(
                logger_config={
                    "type": "ray.tune.logger.TBXLogger",
                    "logdir": config.paths.algo_dir,
                },
                **config.rl_debug,
            )
            .resources(**config.rl_resources)
            .build()
        )

        self.config = config
        self.action_map = build_action_map(config.trader)
        self.checkpoint = ""

    def reload_checkpoint(self):
        """Reload checkpoint.

        Looks for the latest RL checkpoint and loads it. This will fail sometimes when
        the agent is being trained simultaneously and the predictor attempts to
        load a partially saved files.

        """
        files = sorted(glob(f"{self.config.paths.algo_dir}/checkpoint*"))
        checkpoint_f = sorted(glob(f"{files[-1]}/*"))
        self.checkpoint = checkpoint_f[0]
        self.trainer.restore(self.checkpoint)

    def predict(self, gym_obs_data):
        """Predict.

        Based on the input gym observation, get the prediction for the normal
        and hedged prediction (hedge = prediction in the same agent that is
        recommending a trade in the opposite direction to your current position.)

        Args:
            gym_obs_data (dict[polars.DataFrame])
                dictionary containing all the feature inputs

        Returns:
            dict(list|float)
                an dict containing the recommended action for the normal, no position
                and hedge cases

        """
        pos_val_hedge = gym_obs_data.pop("pos_val_hedge")
        mask_hedge = gym_obs_data.pop("mask_hedge")

        pos_val_no_pos = gym_obs_data.pop("pos_val_no_pos")
        mask_no_pos = gym_obs_data.pop("mask_no_pos")

        # with proper mask
        act_pos, _, act_dist_pos = self.trainer.compute_single_action(
            gym_obs_data,
            explore=False,
            full_fetch=True,
        )

        # no pos
        gym_obs_data["mask"] = mask_no_pos
        gym_obs_data["pos_val"] = pos_val_no_pos
        act_no_pos, _, act_dist_no_pos = self.trainer.compute_single_action(
            gym_obs_data,
            explore=False,
            full_fetch=True,
        )

        # hedge
        gym_obs_data["mask"] = mask_hedge
        gym_obs_data["pos_val"] = pos_val_hedge
        act_hedge, _, act_dist_hedge = self.trainer.compute_single_action(
            gym_obs_data,
            explore=False,
            full_fetch=True,
        )

        out = {
            "action_pos": int(act_pos),
            "action_no_pos": int(act_no_pos),
            "action_hedge": int(act_hedge),
            "action_dist_pos": act_dist_pos["action_dist_inputs"].tolist(),
            "action_dist_no_pos": act_dist_no_pos["action_dist_inputs"].tolist(),
            "action_dist_hedge": act_dist_hedge["action_dist_inputs"].tolist(),
            "checkpoint": self.checkpoint,
        }
        return out
