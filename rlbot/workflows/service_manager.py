"""Start and stop external apps.

Utility function for starting and stopping other open source applications:
- MetaTrader5
- ray
- Aerospike
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
from glob import glob
from pathlib import Path
from time import sleep

import ray
import requests

from rlbot.utils.configs.constants import mt5_api_port_map
from rlbot.utils.configs.constants import mt5_creds
from rlbot.utils.configs.constants import root_dir
from rlbot.utils.logging import get_logger

logger = get_logger(__name__, log_level=logging.INFO)


def start_process(cmd_str, blocking=True):
    """Start process.

    Args:
        cmd_str (str):
            bash command to run as a single string, i.e. 'asinfo -v STATUS'
        blocking (bool):
            True is the process should block the current terminal, otherwise
            process is opened in background

    Returns:
        process object

    """
    cmd_str = shlex.split(cmd_str)
    p = subprocess.Popen(
        cmd_str,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
        shell=False,
        # preexec_fn=os.setsid,
    )
    if blocking:
        p.wait()
    else:
        return p


def get_pids(name):
    """Get process ids.

    Gets the process IDs based on the process name. Process name must be an exact match.
    See running processes by first running 'ps -A' to list all processes.

    Args:
        name (str):
            name of the process

    Returns:
        list:
            process ids

    """
    try:
        return list(map(int, subprocess.check_output(["pgrep", name]).split()))
    except Exception:
        return []


def kill_processes(pids):
    """Kills process.

    Kills process based on process ids

    Args:
        pids (list):
            list of process ids

    Returns:
        None

    """
    if len(pids) > 0:
        for pid in pids:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                logger.info(f"process id: {pid} killed")
            except Exception:
                logger.info(f"process id: {pid} already killed")


def start_aerospike():
    """Start Aerospike."""
    os.makedirs(f"{root_dir}/data/aerospike", exist_ok=True)
    res = subprocess.run("asinfo -v STATUS", capture_output=True, text=True, shell=True)
    if "ERROR" in res.stdout != "":
        cmd_str = "asd --config-file ./infrastructure/aerospike/aerospike.conf"
        _ = start_process(cmd_str, blocking=False)
        logger.info("Aerospike started")
    else:
        logger.info("Aerospike already started")


def stop_aerospike():
    """Stop Aerospike."""
    pids = get_pids("asd")
    kill_processes(pids)
    logger.info("Aerospike stopped")


def stop_python():
    """Stop Python."""
    pids = get_pids("python.exe")
    kill_processes(pids)
    pids = get_pids("python")
    kill_processes(pids)
    logger.info("Python stopped")


def start_mt5():
    """Start MetaTrader5."""
    cmd_str = 'wine "/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"'
    logger.debug(cmd_str)
    _ = start_process(cmd_str, blocking=False)
    logger.info("MT5 started")


def stop_mt5():
    """Stop MetaTrader5."""
    pids = get_pids("terminal64.exe")
    kill_processes(pids)
    logger.info(f"MetaTrader5 stopped: process ids {pids} killed")


def start_ray():
    """Start ray."""
    try:
        ray.init(address="auto")
        assert ray.is_initialized()
        logger.info("Ray already started")
    except ConnectionError:
        import tensorflow as tf

        num_cpus = os.cpu_count()
        num_gpus = len(tf.config.list_physical_devices("GPU"))

        # TODO should I be specifying the ports?
        cmd_str = f"ray start --head --num-cpus={num_cpus} --num-gpus={num_gpus} "
        logger.debug(cmd_str)
        _ = start_process(cmd_str, blocking=False)

        sleep(15)
        ray.init(address="auto")
        assert ray.is_initialized(), "Ray failed to initialize"

        logger.info("Ray started")


def start_mt5_api(broker, symbol):
    """Start MetaTrader5 api."""
    cmd_str = f"wine poetry run python ./apis/mt5.py -b {broker} -s {symbol}"
    logger.debug(cmd_str)
    _ = start_process(cmd_str, blocking=False)
    mt5_config = mt5_creds[broker]["demo"]

    mt5_folder = str(Path(mt5_config["path"]).parents[1])

    # Copy from 0 to N (need a terminal per broker)
    if not os.path.isdir(mt5_folder):
        mt5_src = str(Path(mt5_folder).parent) + "/0"
        _ = shutil.copytree(mt5_src, mt5_folder)

    port = mt5_api_port_map[broker][symbol]
    connected = False
    while not connected:
        try:
            _ = requests.post(f"http://127.0.0.1:{port}/init", json=mt5_config)
            _ = requests.get(f"http://127.0.0.1:{port}/healthcheck").json()
            connected = True
        except Exception:
            sleep(1)
    logger.info(f"MT5 api started for {broker} {symbol}")


def start_all_mt5_apis():
    """Start MetaTrader5 api."""
    for broker, port_map in mt5_api_port_map.items():
        for symbol, _ in port_map.items():
            _ = start_mt5_api(broker, symbol)


def start_redis():
    """Start redis.

    Default address is localhost:6369

    """
    cmd_str = "redis-server infrastructure/redis/redis.conf"
    logger.debug(cmd_str)
    _ = start_process(cmd_str, blocking=False)
    logger.info("redis started")


def stop_redis():
    """Start redis."""
    cmd_str = "redis-cli -p 6369 shutdown"
    logger.debug(cmd_str)
    _ = start_process(cmd_str, blocking=False)
    logger.info("redis started")


def start_tensorboard():
    """Start tensorboard.

    Default address is http://localhost:6006/


    """
    folder = glob(f"{os.getcwd()}/data/agent/*")
    cmd_str = "tensorboard --logdir_spec="
    for f in folder:
        cmd_str += f"{f.split('/')[-1]}:{f}/algo,"
    cmd_str = cmd_str[:-1]
    _ = start_process(cmd_str, blocking=False)
    logger.info("tensorboard started")


def stop_tensorboard():
    """Stop tensorboard."""
    pids = get_pids("tensorboard")
    kill_processes(pids)
    logger.info("tensorboard stopped")


def stop_ray():
    """Stop ray."""
    _ = start_process("ray stop", blocking=False)
    logger.info("Ray stopped")


def start_services():
    """Start all services."""
    start_aerospike()
    start_ray()
    start_tensorboard()
    start_redis()


def stop_services():
    """Stop all services."""
    stop_aerospike()
    stop_mt5()
    stop_ray()
    stop_tensorboard()
    stop_redis()
