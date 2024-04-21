from __future__ import annotations

from rlbot.gym_env.metrics import TradingMetrics


def test_trading_metrics():
    metrics = TradingMetrics()

    metrics.update_metrics(1.0, 10.0)

    metrics.decide_repeat(
        0.5,  # win_rate_thresh,
        -10.0,  # max_loss_thresh,
        30.0,  # drawdown_thresh,
        0.0,  # end_cum_reward_thresh,
        0.0,  # min_cum_reward_thresh,
    )

    assert not metrics.repeat_ep

    metrics.reset_metrics(100)
    metrics.update_metrics(-1.0, -10.0)
    metrics.decide_repeat(
        0.5,  # win_rate_thresh,
        -10.0,  # max_loss_thresh,
        30.0,  # drawdown_thresh,
        0.0,  # end_cum_reward_thresh,
        0.0,  # min_cum_reward_thresh,
    )

    assert metrics.repeat_ep
