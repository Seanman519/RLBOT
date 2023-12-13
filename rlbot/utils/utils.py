"""Miscellaneous utility functions."""
from __future__ import annotations


def add_underscores(num):
    """Adds underscores to a number every third digit.

    Args:
        num (int or str):
            The number to add underscores to.

    Returns:
        str:
            The number with underscores added.
    """
    num_str = str(num)  # convert the number to a string
    num_len = len(num_str)
    underscored_num = ""
    for i in range(num_len):
        underscored_num += num_str[i]
        if (num_len - i - 1) % 3 == 0 and i != num_len - 1:
            underscored_num += "_"
    return underscored_num
