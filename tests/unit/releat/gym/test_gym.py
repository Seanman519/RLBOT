from __future__ import annotations

import numpy as np

from rlbot.gym_env.gym_env import FxEnv
from rlbot.utils.configs.config_builder import load_config


config = load_config("t00001", enrich_feat_spec=True, is_training=False)


def test_gym_env():
    env = FxEnv(dict(config))
    obs, _ = env.reset()
    mask = obs["mask"]
    for _ in range(300):
        action_choices = np.where(mask == 1)[0]
        action = np.random.choice(action_choices)
        obs, _, done, _, _ = env.step(action)
        mask = obs["mask"]
        # print(obs['pos_val'], ' | ', reward,' | ', obs['mask'])
        if done:
            break
    res, cp = env.get_results()
    assert len(res) > 0
    assert len(cp) > 0
