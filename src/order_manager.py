import logging
import math
import pandas as pd
import numpy as np


def check_main_exit_conditions(order, row, current_bar_index, now_timestamp):
    """Check exit conditions for an order."""
    from src import strategy as _s
    MAX_HOLDING_BARS = getattr(_s, "MAX_HOLDING_BARS", 0)

    order_closed_this_bar = False
    exit_price_final = np.nan
    close_reason_final = "UNKNOWN_EXIT"
    close_timestamp_final = now_timestamp

    side = order.get("side")
    sl_price_order = pd.to_numeric(order.get("sl_price"), errors="coerce")
    tp_price_order = pd.to_numeric(order.get("tp_price"), errors="coerce")
    entry_price_order = pd.to_numeric(order.get("entry_price"), errors="coerce")

    current_high = pd.to_numeric(getattr(row, "High", np.nan), errors="coerce")
    current_low = pd.to_numeric(getattr(row, "Low", np.nan), errors="coerce")
    current_close = pd.to_numeric(getattr(row, "Close", np.nan), errors="coerce")
    be_triggered = order.get("be_triggered", False)
    entry_bar_count_order = order.get("entry_bar_count")
    entry_time_log = order.get("entry_time", "N/A")

    price_tolerance = 0.05

    sl_text = f"{sl_price_order:.5f}" if pd.notna(sl_price_order) else "NaN"
    tp_text = f"{tp_price_order:.5f}" if pd.notna(tp_price_order) else "NaN"
    logging.debug(
        f"            [Exit Check V2.1] Order {entry_time_log} "
        f"Side: {side}, SL: {sl_text}, TP: {tp_text}, BE: {be_triggered}"
    )
    logging.debug(
        f"            [Exit Check V2.1] Bar Prices: H={current_high:.5f}, L={current_low:.5f}, C={current_close:.5f}"
    )

    if be_triggered and pd.notna(sl_price_order) and pd.notna(entry_price_order) and math.isclose(
        sl_price_order, entry_price_order, abs_tol=price_tolerance
    ):
        if side == "BUY" and (current_low <= sl_price_order + price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "BE-SL"
            exit_price_final = sl_price_order
        elif side == "SELL" and (current_high >= sl_price_order - price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "BE-SL"
            exit_price_final = sl_price_order

    if not order_closed_this_bar and pd.notna(sl_price_order):
        if side == "BUY" and (current_low <= sl_price_order + price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "SL"
            exit_price_final = sl_price_order
        elif side == "SELL" and (current_high >= sl_price_order - price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "SL"
            exit_price_final = sl_price_order

    if not order_closed_this_bar and pd.notna(tp_price_order):
        if side == "BUY" and (current_high >= tp_price_order - price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "TP"
            exit_price_final = tp_price_order
        elif side == "SELL" and (current_low <= tp_price_order + price_tolerance):
            order_closed_this_bar = True
            close_reason_final = "TP"
            exit_price_final = tp_price_order

    if not order_closed_this_bar:
        if entry_bar_count_order is not None:
            bars_held = current_bar_index - entry_bar_count_order
            if bars_held >= MAX_HOLDING_BARS:
                if pd.notna(current_close):
                    exit_price_final = current_close
                    close_reason_final = f"MaxBars ({MAX_HOLDING_BARS})"
                else:
                    exit_price_final = sl_price_order if pd.notna(sl_price_order) else entry_price_order
                    if pd.isna(exit_price_final):
                        exit_price_final = 0
                    close_reason_final = f"MaxBars ({MAX_HOLDING_BARS})_CloseNaN"
                order_closed_this_bar = True
        return order_closed_this_bar, exit_price_final, close_reason_final, close_timestamp_final

    return order_closed_this_bar, exit_price_final, close_reason_final, close_timestamp_final


def update_open_order_state(
    order,
    current_high,
    current_low,
    current_atr,
    avg_atr,
    now,
    base_be_r_thresh,
    fold_sl_multiplier_base,
    base_tp_multiplier_config,
    be_sl_counter,
    tsl_counter,
):
    """Update open order state for BE and TSL logic."""
    from src import strategy as _s

    DYNAMIC_BE_ATR_THRESHOLD_HIGH = getattr(_s, "DYNAMIC_BE_ATR_THRESHOLD_HIGH", 0)
    DYNAMIC_BE_R_ADJUST_HIGH = getattr(_s, "DYNAMIC_BE_R_ADJUST_HIGH", 0)
    ADAPTIVE_TSL_START_ATR_MULT = getattr(_s, "ADAPTIVE_TSL_START_ATR_MULT", 1.0)
    update_breakeven_half_tp = getattr(_s, "update_breakeven_half_tp")
    update_tsl_only = getattr(_s, "update_tsl_only")
    compute_trailing_atr_stop = getattr(_s, "compute_trailing_atr_stop")
    update_trailing_tp2 = getattr(_s, "update_trailing_tp2")
    dynamic_tp2_multiplier = getattr(_s, "dynamic_tp2_multiplier")

    be_triggered_this_bar = False
    tsl_updated_this_bar = False
    order_side = order.get("side")
    entry_price = pd.to_numeric(order.get("entry_price"), errors="coerce")
    original_sl_price = pd.to_numeric(order.get("original_sl_price"), errors="coerce")
    current_sl_price_in_order = pd.to_numeric(order.get("sl_price"), errors="coerce")
    atr_at_entry = pd.to_numeric(order.get("atr_at_entry"), errors="coerce")
    entry_time_log = order.get("entry_time", "N/A")

    order, be_half = update_breakeven_half_tp(order, current_high, current_low, now)
    if be_half:
        be_sl_counter += 1
        be_triggered_this_bar = True

    if not order.get("be_triggered", False):
        dynamic_be_r_threshold = base_be_r_thresh
        try:
            current_atr_for_be_calc = pd.to_numeric(atr_at_entry, errors="coerce")
            current_avg_atr_for_be_calc = pd.to_numeric(avg_atr, errors="coerce")
            if (
                pd.notna(current_atr_for_be_calc)
                and pd.notna(current_avg_atr_for_be_calc)
                and not np.isinf(current_atr_for_be_calc)
                and not np.isinf(current_avg_atr_for_be_calc)
                and current_avg_atr_for_be_calc > 1e-9
                and (current_atr_for_be_calc / current_avg_atr_for_be_calc) > DYNAMIC_BE_ATR_THRESHOLD_HIGH
            ):
                dynamic_be_r_threshold += DYNAMIC_BE_R_ADJUST_HIGH
        except Exception:
            logging.warning(f"(Warning) Error calculating dynamic BE threshold at {now}")

        if dynamic_be_r_threshold > 0:
            if pd.notna(entry_price) and pd.notna(original_sl_price):
                sl_delta_price_be = abs(entry_price - original_sl_price)
                if sl_delta_price_be > 1e-9:
                    be_trigger_price_diff = sl_delta_price_be * dynamic_be_r_threshold
                    be_trigger_price = (
                        entry_price + be_trigger_price_diff if order_side == "BUY" else entry_price - be_trigger_price_diff
                    )
                    trigger_hit = False
                    if order_side == "BUY" and current_high >= be_trigger_price:
                        trigger_hit = True
                    elif order_side == "SELL" and current_low <= be_trigger_price:
                        trigger_hit = True
                    if trigger_hit and not math.isclose(
                        current_sl_price_in_order if pd.notna(current_sl_price_in_order) else -np.inf,
                        entry_price if pd.notna(entry_price) else np.inf,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        order["sl_price"] = entry_price
                        order["be_triggered"] = True
                        order["be_triggered_time"] = now
                        be_sl_counter += 1
                        be_triggered_this_bar = True

    if not be_triggered_this_bar:
        if not order.get("tsl_activated", False) and pd.notna(atr_at_entry) and atr_at_entry > 1e-9:
            tsl_activation_price_diff = ADAPTIVE_TSL_START_ATR_MULT * atr_at_entry
            tsl_activation_price = entry_price + tsl_activation_price_diff if order_side == "BUY" else entry_price - tsl_activation_price_diff
            if (order_side == "BUY" and current_high >= tsl_activation_price) or (
                order_side == "SELL" and current_low <= tsl_activation_price
            ):
                order["tsl_activated"] = True
                if order_side == "BUY":
                    order["peak_since_tsl_activation"] = current_high
                else:
                    order["trough_since_tsl_activation"] = current_low
        if order.get("tsl_activated"):
            tsl_atr_mult_fixed = 1.5
            order, sl_updated_flag = update_tsl_only(
                order,
                current_high,
                current_low,
                current_atr,
                avg_atr,
                atr_multiplier=tsl_atr_mult_fixed,
            )
            if sl_updated_flag:
                tsl_updated_this_bar = True
                tsl_counter += 1
            new_sl_atr = compute_trailing_atr_stop(
                entry_price,
                current_high if order_side == "BUY" else current_low,
                current_atr,
                order_side,
                order.get("sl_price"),
            )
            new_sl_val = pd.to_numeric(new_sl_atr, errors="coerce")
            current_sl_val = pd.to_numeric(order.get("sl_price"), errors="coerce")
            if pd.notna(new_sl_val) and pd.notna(current_sl_val):
                if (order_side == "BUY" and new_sl_val > current_sl_val) or (
                    order_side == "SELL" and new_sl_val < current_sl_val
                ):
                    order["sl_price"] = new_sl_val
                    tsl_updated_this_bar = True

    tp2_mult = dynamic_tp2_multiplier(current_atr, avg_atr, base=base_tp_multiplier_config)
    order = update_trailing_tp2(order, current_atr, tp2_mult)
    return order, be_triggered_this_bar, tsl_updated_this_bar, be_sl_counter, tsl_counter
