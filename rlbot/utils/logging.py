"""Logging.

Utility logging function to help with info, debug and warning
messages.

"""
from __future__ import annotations

import logging

import typer


class TyperLoggerHandler(logging.Handler):
    """Logger Handler."""

    def emit(self, record: logging.LogRecord) -> None:
        """Color and print text."""
        fg = None
        bg = None
        if record.levelno == logging.DEBUG:
            fg = typer.colors.BLACK
        elif record.levelno == logging.INFO:
            fg = typer.colors.BRIGHT_BLUE
        elif record.levelno == logging.WARNING:
            fg = typer.colors.BRIGHT_MAGENTA
        elif record.levelno == logging.CRITICAL:
            fg = typer.colors.BRIGHT_RED
        elif record.levelno == logging.ERROR:
            fg = typer.colors.BRIGHT_WHITE
            bg = typer.colors.RED
        typer.secho(self.format(record), bg=bg, fg=fg)


def get_logger(name, log_level=logging.INFO):
    """Get logger."""
    formatter = logging.Formatter(
        "%(asctime)s  %(levelname)-5s  %(name)-5s  |  %(message)s",
    )
    typer_handler = TyperLoggerHandler()
    typer_handler.setFormatter(formatter)
    logging.basicConfig(level=log_level, handlers=(typer_handler,))
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    return logger
