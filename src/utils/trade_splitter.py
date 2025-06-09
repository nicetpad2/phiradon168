import logging

# Use a local logger to prevent circular import with ``src.config`` during test
# collection.
logger = logging.getLogger(__name__)

import os
import pandas as pd


def normalize_side(side: object) -> str:
    """[Patch v5.7.4] Normalize side value to BUY/SELL/NORMAL."""
    val = str(side).strip().upper()
    if val in {"BUY", "B", "1"}:
        return "BUY"
    if val in {"SELL", "S", "0", "-1"}:
        return "SELL"
    return "NORMAL"


def has_buy_sell(df: pd.DataFrame) -> bool:
    """[Patch v5.7.4] Return True if any BUY/SELL rows detected."""
    sides = df.get("side")
    if sides is None:
        return False
    sides = sides.astype(str).str.strip().str.upper()
    return sides.isin({"BUY", "SELL", "B", "S", "1", "0", "-1"}).any()


def split_trade_log(df: pd.DataFrame, output_dir: str) -> None:
    """[Patch v5.7.4] Split trade log by side with detailed logging."""
    os.makedirs(output_dir, exist_ok=True)
    buy_rows, sell_rows, normal_rows = [], [], []
    for idx, trade in df.iterrows():
        side = normalize_side(trade.get("side", ""))
        if side == "BUY":
            buy_rows.append(trade)
            msg = f"เขียนไฟล์ trade_log_BUY.csv เลขแถวที่ {idx}"
            logger.info(msg)
            logging.getLogger().info(msg)
        elif side == "SELL":
            sell_rows.append(trade)
            msg = f"เขียนไฟล์ trade_log_SELL.csv เลขแถวที่ {idx}"
            logger.info(msg)
            logging.getLogger().info(msg)
        else:
            normal_rows.append(trade)
            msg = f"เขียนไฟล์ trade_log_NORMAL.csv เลขแถวที่ {idx}"
            logger.info(msg)
            logging.getLogger().info(msg)

    written = 0
    if buy_rows:
        pd.DataFrame(buy_rows).to_csv(os.path.join(output_dir, "trade_log_BUY.csv"), index=False)
        written += 1
    if sell_rows:
        pd.DataFrame(sell_rows).to_csv(os.path.join(output_dir, "trade_log_SELL.csv"), index=False)
        written += 1
    if normal_rows:
        pd.DataFrame(normal_rows).to_csv(os.path.join(output_dir, "trade_log_NORMAL.csv"), index=False)
        written += 1
    if written == 0:
        msg = "[QA-WARNING] ไม่พบรายการใดถูกเขียนลง trade_log_BUY.csv / SELL.csv / NORMAL.csv  กรุณาตรวจสอบเงื่อนไขในโค้ด"
        logger.warning(msg)
        logging.getLogger().warning(msg)
