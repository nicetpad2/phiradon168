"""Main entry point for phiradon168 strategy."""

import logging
from .data_loader import load_data
from .strategy import simple_strategy


def run():
    df_m1, _ = load_data()
    buys, sells = simple_strategy(df_m1)
    logging.info(f"Executed simple strategy: buys={buys}, sells={sells}")


if __name__ == "__main__":
    run()
