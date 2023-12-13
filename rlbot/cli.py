"""CLI for rlbot."""
from __future__ import annotations

from time import sleep

import aerospike
import typer

from rlbot.data.pipeline import populate_train_data
from rlbot.data.pipeline import update_gym_env_hparam
from rlbot.utils.configs.config_builder import load_config
from rlbot.utils.logging import get_logger
from rlbot.workflows import generate_signals
from rlbot.workflows import service_manager as sm
from rlbot.workflows.service_manager import start_services
from rlbot.workflows.service_manager import stop_services
from rlbot.workflows.train_rl_agent import train_rl_agent

app = typer.Typer()


logger = get_logger(__name__)


@app.command()
def start():
    """Start all services."""
    start_services()


@app.command()
def stop():
    """Stop all services."""
    stop_services()


@app.command()
def launch_mt5_api(broker, symbol):
    """Start mt5 api for one broker."""
    sm.start_mt5_api(broker, symbol)


@app.command()
def launch_all_mt5_apis():
    """Starts mt5 apis for all brokers."""
    sm.start_all_mt5_apis()
    sleep(30)


@app.command()
def build_train_data(agent_version):
    """Builds training data."""
    config = load_config(agent_version)
    _ = populate_train_data(config, mode="initialise")


@app.command()
def launch_train_data_updater(agent_version):
    """Periodically updates train data db."""
    logger.info(f"WIP for {agent_version}")


@app.command()
def train(agent_version):
    """Train RL agent."""
    config, AgentModel = load_config(
        agent_version,
        enrich_feat_spec=True,
        load_model=True,
    )
    client = aerospike.client(config.aerospike.connection).connect()
    _ = update_gym_env_hparam(config, client, overwrite_max_samples=False)
    train_rl_agent(config, AgentModel)


@app.command()
def generate_signal(agent_version):
    """Generate a trading signal."""
    _ = generate_signals.generate_signal(agent_version)


@app.command()
def launch_trader():
    """Launch trading agent."""
    exec(open("./rlbot/trader/trader.py").read())


if __name__ == "__main__":
    app()
