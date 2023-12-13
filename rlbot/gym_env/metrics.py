"""Gym environment metrics."""
from __future__ import annotations

from numba import boolean
from numba import float32
from numba import int32
from numba.experimental import jitclass

spec = [
    ("repeat_num", int32),
    ("wins", int32),
    ("losses", int32),
    ("max_loss", float32),
    ("cum_reward", float32),
    ("min_cum_reward", float32),
    ("peak", float32),
    ("trough", float32),
    ("drawdown", float32),
    ("repeat_ep", boolean),
    ("ep_start_ind", int32),
    ("max_ep_repeats", int32),
]


@jitclass(spec)
class TradingMetrics:
    """Trading Metrics."""

    def __init__(self):
        """Init."""
        self.repeat_num = 0
        self.max_ep_repeats = 1

        _ = self.reset_metrics(-1)

    def reset_metrics(self, start_ind):
        """Reset metrics."""
        self.wins = 0
        self.losses = 0

        self.max_loss = 0.0
        self.cum_reward = 0.0
        self.min_cum_reward = 0.0

        self.peak = 0.0
        self.trough = 0.0
        self.drawdown = 0.0

        self.repeat_ep = False
        self.ep_start_ind = start_ind

    def set_max_repeat_num(self, max_ep_repeats):
        """Set maximum number of repeats."""
        self.max_ep_repeats = max_ep_repeats

    def update_metrics(self, reward, cum_reward):
        """Update metrics."""
        if reward > 0:
            self.wins += 1

        elif reward < 0:
            self.losses += 1

            if reward < self.max_loss:
                self.max_loss = reward

        self.cum_reward = cum_reward

        if cum_reward < self.min_cum_reward:
            self.min_cum_reward = cum_reward

        if cum_reward > self.peak:
            self.peak = cum_reward
            self.trough = cum_reward
        else:
            if cum_reward < self.trough:
                self.trough = cum_reward

        drawdown = self.peak - self.trough
        if drawdown > self.drawdown:
            self.drawdown = drawdown

    def decide_repeat(
        self,
        win_rate_thresh,
        max_loss_thresh,
        drawdown_thresh,
        end_cum_reward_thresh,
        min_cum_reward_thresh,
    ):
        """Decide repeat."""
        win_rate = self.wins / max(self.wins + self.losses, 1)
        # ep_ind_str = add_underscores(self.ep_start_ind).rjust(10)
        ep_ind_str = str(self.ep_start_ind).rjust(10)
        cond_str_len = 14
        val_str_len = 10
        repeat_ep = False
        if win_rate < win_rate_thresh:
            repeat_ep = True
            # string is 15 characters
            val_str = f"{int(win_rate*100)}% ({int(self.wins+self.losses)})"
            val_str = val_str.rjust(val_str_len)
            print(
                (
                    f"{ep_ind_str}: {'win rate'.rjust(cond_str_len)}"
                    f" {val_str} < {str(int(win_rate_thresh*100)).rjust(4)}%"
                ),
            )
        elif self.cum_reward < end_cum_reward_thresh:
            repeat_ep = True
            val_str = str(int(self.cum_reward)).rjust(val_str_len)
            print(
                (
                    f"{ep_ind_str}: {'final reward'.rjust(cond_str_len)}"
                    f" {val_str} < {str(int(end_cum_reward_thresh)).rjust(4)}"
                ),
            )

        elif self.drawdown > drawdown_thresh:
            repeat_ep = True
            val_str = str(int(self.drawdown)).rjust(val_str_len)
            print(
                (
                    f"{ep_ind_str}: {'drawdown'.rjust(cond_str_len)}"
                    f" {val_str} > {str(int(drawdown_thresh)).rjust(4)}"
                ),
            )

        elif self.min_cum_reward < min_cum_reward_thresh:
            repeat_ep = True
            val_str = str(int(self.min_cum_reward)).rjust(val_str_len)
            print(
                (
                    f"{ep_ind_str}: {'min cum reward'.rjust(cond_str_len)}"
                    f" {val_str} < {str(int(min_cum_reward_thresh)).rjust(4)}"
                ),
            )

        elif self.max_loss < max_loss_thresh:
            repeat_ep = True
            val_str = str(int(self.max_loss)).rjust(val_str_len)
            print(
                (
                    f"{ep_ind_str}: {'max trade loss'.rjust(cond_str_len)}"
                    f" {val_str} < {str(int(max_loss_thresh)).rjust(4)}"
                ),
            )

        if repeat_ep & (self.repeat_num < self.max_ep_repeats):
            # only repeat the same episode a max of 5 times
            self.repeat_ep = True
            self.repeat_num += 1
        else:
            self.repeat_ep = False
            self.repeat_num = 0
