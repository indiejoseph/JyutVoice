from typing import List

import hydra
from omegaconf import OmegaConf
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from jyutvoice.utils import pylogger

log = pylogger.get_pylogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    # Convert dict to DictConfig if needed
    if isinstance(callbacks_cfg, dict) and not isinstance(callbacks_cfg, DictConfig):
        callbacks_cfg = OmegaConf.create(callbacks_cfg)

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(
                f"Instantiating callback <{cb_conf._target_}>"
            )  # pylint: disable=protected-access
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig or dict object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    # Convert dict to DictConfig if needed
    if isinstance(logger_cfg, dict) and not isinstance(logger_cfg, DictConfig):
        logger_cfg = OmegaConf.create(logger_cfg)

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig or dict!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(
                f"Instantiating logger <{lg_conf._target_}>"
            )  # pylint: disable=protected-access
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
