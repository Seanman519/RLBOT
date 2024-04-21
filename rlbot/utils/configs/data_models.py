"""Data Models.

All data models for configs

"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Paths(BaseModel):
    """Paths for data storage."""

    # base folder is typically the repo directory
    root_dir: str
    # folder for tick data
    tick_data_dir: str
    # folder for storing training artifacts, such as model, training checkpoints, etc.
    algo_dir: str
    # folder for storing training evaluation metrics
    eval_dir: str
    # folder for storing intermediate files used to generate the initial features
    feature_dir: str
    # folder for storing the incremental features used to update the database
    update_tick_data_dir: str
    # folder for collecting logs from the execution agent
    trader_log_dir: str


class SymbolSpec(BaseModel):
    """Trading instrument information."""

    # trading instrument, i.e. EURUSD
    symbol: str
    # value of a pip
    pip: float
    # contract size
    contract_size: int
    # currency of profit / loss
    currency: str
    # tick value per lot
    pip_val: float
    # round trip comission in usd
    commission: float
    # broker
    broker: str
    # maintenance - amount required to carry past daily close
    maintenance: float
    # margin - amount required to open a position
    margin: float


class RawDataConfig(BaseModel):
    """Parameters for processing the raw tick data."""

    # if None, then load all available tick data files. Otherwise put in dates, i.e.
    # ['2023-04-01,'2023-05-01'] to load a specific range of ticks
    tick_file_dates: list[str] | None = None
    # polars format for how frequently the trading agent can make a decision, i.e. 10s
    # means that an agent makes a decision every 10 seconds
    trade_timeframe: str
    # number of seconds that off the trade_timeframe to make decision. A trade_timeframe
    # =10s and a trade_time_offset=3s means that the agent trades on the 13th, 23rd, 33rd
    # 43rd and 53rd second of each minute
    trade_time_offset: str
    # clip time difference, important to avoid outliers for time differencing, especially
    # for weekends, market close periods
    tick_time_diff_clip_val: int
    # simulated inference time, when we take an open or close action, we take the symbol
    # price that is min_inf_time_s to max_inf_time_s after the current time
    min_inf_time_s: str
    max_inf_time_s: str
    # min and max values for each gym observation
    min_obs_val: float
    max_obs_val: float
    # live or demo data
    data_mode: str


class GymEnvConfig(BaseModel):
    """Gym hyperparameters.

    Programmtic method of changing the gym environment without restarting the
    training function
    Controls hyperparameters for:
    - different start / end points for each episode
    - reward criteria for increasing / decreasing data sampling range

    These are saved in aerospike so we can dynamically change without stopping
    the training process. Other notes:
    - all field names must be 14 characters or less due to aerospike bin name limit
    - suffix of '_t' indicates a threshold
    - suffix of '_p indicates a probability (ranging from 0-1)
    """

    # Gym hyperparameters to control trades and episode lengths

    # period in seconds for reloading training hyper paramemters from aerospike
    hp_reload_t_s: int
    # max steps per episode
    max_ep_step: int
    # skip steps to make rewards more frequent
    skip_step: int
    # min and max number of steps to hold a position
    min_hold_t: int
    max_hold_t: int
    # max trades per episode
    max_trades: int
    # stop value to close position in pips - must be negative number
    stop_val: int
    # min and max rewards for ending an episode
    min_ep_r: int
    max_ep_r: int
    # commission in pips per trade
    commission: float
    # penalty for each step (scaled to the gym reward)
    step_penalty: float
    # penalty for holding a position
    hold_penalty: float
    # evaluation length, used to exclude data from the agent training process
    eval_len: int
    # if not is_training, then we use the last eval_len steps to evaluate the
    # agent and log all the actions and positions
    is_training: bool
    log_actions: bool

    # criteria for increasing number of training records

    # number of training records
    rec_num: int
    # how many to increase the number of training records by
    rec_growth: float
    # ignore the first X episodes when checking whether agent reward is sufficient
    rec_warm_up: int
    # number of episodes that reward must be higher than the threshold
    rec_ep_t: int
    # if the episode is higher than this threshold, increase the number of steps
    rec_reward_t: float

    # criteria to repeat the episode or sample starting point

    # max times to repeat an episode that meets the following criteria
    max_ep_repeats: int
    # trading win rate is below X%
    win_rate_t: float
    # maximum loss in one position
    pos_loss_t: float
    # maximum drawdown in the episode
    drawdown_t: float
    # cumulative reward at the end of the episode
    end_cum_r_t: float
    # min cumulative reward threshold, aka lowest score during whole episode
    min_cum_r_t: float
    # over sample num - higher probability to start in the X most recent samples
    osample_num: int
    # if a random number is higher the the osample_prob, then sample from the
    # osample_num most recent samples
    osample_p: float

    # criteria for changing trade conditions

    # criteria to limit the agent's action space
    # extra action masking based on position size - i.e. agent can only sometimes trade
    # up to 1, 2 or 3 lots
    mask_size_p: float
    # double sided probability of extra action masking based on position direction, i.e.
    # agent can sometimes only long or short for an episode
    mask_dir_p: float

    # agent training hyperparameters

    # how many times to run the .train()
    train_iter: int
    # how frequently to save checkpoint
    save_freq: int


class AerospikeConfig(BaseModel):
    """Aerospike config."""

    # connection parameters
    connection: dict
    # namespace (either dev or prod)
    namespace: str
    # usually the same as agent_version
    set_name: str


class MT5Config(BaseModel):
    """MT5 config."""

    server: str
    login: int
    password: str
    path: str


class RedisConfig(BaseModel):
    """Redis config."""

    host: str
    port: int


class TransformerConfig(BaseModel):
    """Transformer config.

    Specifies transformation function for each feature.

    """

    # name of the transformation - must be in the predefined list
    name: str
    # variant or method for a particular transformation
    method: str

    # parameters that are relevant for clip
    upper_lim: Any | None = None
    lower_lim: Any | None = None
    scale_factor: Any | None = None

    # parameters relevant for Power transformer
    lam: Any | None = None
    mean: Any | None = None
    std: Any | None = None

    # parameters for clipping
    clip_min: Any | None = None
    clip_max: Any | None = None

    # parameters that are relevant for scale, first / last difference is an example
    # of where is_elementwise would be true whilst tick count is example of where it
    # would be false
    is_elementwise: bool | None = None


class SimpleFeatureConfig(BaseModel):
    """Feature config.

    Simple feature is defined as:
    - one trading instrument
    - one broker
    - calculation based on tick data
    - using one or more attributes of the tick data, i.e. last price and flags

    #TODO complex features which are built of candle data or multple simple feature

    """

    # name of feature calculation - must be in a predefined list
    name: str
    # broker to allow for one model to process data from different brokers
    broker: str
    # index of the feature, starting from 0 to n
    index: int
    # symbol is trading instrument
    symbol: str
    # timeframe using polars format, i.e. 10m for 10 minute timeframe for each feature
    timeframe: str
    # list of tick data attributes required for the feature, i.e. last price and flag
    inputs: list[str]
    # output shape is usually (number of timesteps for model input,number of ouputs from
    # feature calculation)
    output_shape: tuple
    # timeframe mode can either be full, partial or rolling - full and partial are not
    # implemented yet
    timeframe_mode: str
    # other feature hyperparameters, for example when calculating skew we can specify
    # minimum number of samples
    kwargs: dict
    # fill nulls with zero or ffill or arbitrary value
    fillna: str
    # transforms applied to each feature
    transforms: list[TransformerConfig]


class FeatureGroupConfig(BaseModel):
    """Feature group config.

    A group of features with the same timeframe. Note that in the feature_config.py
    file of each experiment.

    Must have:
    - all features must be in the same timeframe for one group
    - each feature name in a group must be unique (i think?)

    """

    # timeframe of the group
    timeframe: str
    # index of the feature group from 0 to n
    index: int
    # how to stack each feature, the default is axis=1 for 2D features, i.e. where
    # timesteps are the rows and columns are the different features
    stack_axis: int
    # a list of simple feature configs
    simple_features: list[SimpleFeatureConfig]


class ModelConfig(BaseModel):
    """Model Config.

    Neural network architecture is specified in each experiment in the model.py file

    Did not want to limit the possible designs for now.

    #TODO make components for neural network models which can be applied in this
    section
    """

    # a dictionary of tuples
    input_shape: dict


class PositionConfig(BaseModel):
    """Position Config.

    Specifies one tradeable slot in the portfolio.

    """

    # symbol is trading instrument
    symbol: str
    # index of the symbol from 0 to n (used to indentify the trading instrument in the
    # gym environment)
    symbol_index: int
    # value of a pip, i.e. 1e-4
    pip_val: float
    # maximum number of long and short position spossible for that trading instrument
    max_long: int
    max_short: int


class TraderConfig(BaseModel):
    """Trading config.

    Configurations when live or paper trading

    """

    # either 'live' or 'paper'
    trade_mode: str
    # list of slots / positions that the agent can trade with
    portfolio: list[PositionConfig]
    # deviation in price for a trade (in points) - See MT5
    deviation: int
    # increments of 0.01
    lot: float


class AgentConfig(BaseModel):
    """Agent config.

    A single class that contains all the configs

    """

    # agent version
    agent_version: str

    # paths
    paths: Paths

    # RL algorithm
    rl_algorithm: str
    # Redis
    redis: RedisConfig
    # Currently only mt5 but in the future ibkr, binance
    platform: str
    # each feature
    # Dict of SymbolSpec
    symbol_info: list[SymbolSpec]
    symbol_info_index: dict
    # Shape of the observation / model input data
    observation_space: Any
    # Shape of the action space
    action_space: Any
    # aerospace login details
    aerospike: AerospikeConfig
    # MT5 login details
    mt5: MT5Config

    raw_data: RawDataConfig
    features: list[FeatureGroupConfig]
    # Gym hyperparams
    # these parameters are uploaded to aerospike. the train process can update these
    # values and the individual gym environments reads off aerospike whilst training
    gym_env: GymEnvConfig
    # Config that ensembles multiple predictions
    trader: TraderConfig

    # rllib specific parameters
    rl_train: dict
    rl_env: dict
    rl_framework: dict
    rl_rollouts: dict
    rl_explore: dict
    rl_reporting: dict
    rl_debug: dict
    rl_resources: dict
