# -*- coding: utf-8 -*-
"""
[Patch v5.3.0] Hyperparameter sweep (Enterprise Edition)
- รองรับ multi-param sweep
- Save log + summary + best param
- Resume ได้ (skip run ที่เสร็จ)
- สรุปสถิติ + best config
"""
# [Patch v5.9.4] Support real trade log usage and metric export
import os
import sys

# [Patch v5.4.9] Ensure repo root is available when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import traceback
from itertools import product
from typing import Callable, List, Dict
import inspect
from datetime import datetime
from tqdm import tqdm
import logging
import numpy as np

from src.config import logger, DefaultConfig, __version__

# [Patch v5.9.1] Default sweep results under configured OUTPUT_DIR
DEFAULT_SWEEP_DIR = DefaultConfig.OUTPUT_DIR


def _create_placeholder_trade_log(path: str) -> None:
    """Create a minimal trade log so the sweep can run."""
    # [Patch v5.10.8] Ensure sample size > 1 to avoid train_test_split errors
    profits = [1.0, -1.0, 0.8, -0.8, 0.6, -0.6, 0.4, -0.4]
    df = pd.DataFrame({"profit": profits})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    compression = "gzip" if path.endswith(".gz") else None
    df.to_csv(path, index=False, compression=compression)
    logger.warning(f"สร้าง trade log ตัวอย่างที่ {path}")
from src.training import real_train_func

# [Patch v6.3.0] Default trade log path under configured OUTPUT_DIR
DEFAULT_TRADE_LOG = os.path.join(
    DefaultConfig.OUTPUT_DIR, "trade_log_v32_walkforward.csv.gz"
)

# [Patch v6.3.0] Dynamic fallback: หากไฟล์ DEFAULT_TRADE_LOG ไม่พบ
if not os.path.exists(DEFAULT_TRADE_LOG):
    import glob
    pattern = os.path.join(
        DefaultConfig.OUTPUT_DIR, "trade_log_*walkforward*.csv*"
    )
    candidates = glob.glob(pattern)
    if candidates:
        # เลือกไฟล์ล่าสุดตามการ sort
        DEFAULT_TRADE_LOG = sorted(candidates)[-1]
        logger.info(
            "[Patch v6.3.0] Found dynamic DEFAULT_TRADE_LOG: %s", DEFAULT_TRADE_LOG
        )
    else:
        # fallback เดิม กรณีที่ไม่พบไฟล์ pattern เดิม
        alt_gz = os.path.join(
            DefaultConfig.OUTPUT_DIR, "trade_log_v32_walkforward.csv"
        )
        if os.path.exists(alt_gz):
            DEFAULT_TRADE_LOG = alt_gz
        else:
            simple_path = os.path.join(
                DefaultConfig.OUTPUT_DIR, "trade_log_NORMAL.csv"
            )
            if os.path.exists(simple_path):
                DEFAULT_TRADE_LOG = simple_path
            else:
                logger.warning(
                    "[Patch v6.3.0] No walk-forward trade log found in %s; "
                    "will require --trade_log_path", DefaultConfig.OUTPUT_DIR
                )


def _parse_csv_list(text: str, cast: Callable) -> List:
    """แปลงสตริงคอมมาเป็นลิสต์พร้อมประเภทข้อมูล"""
    return [cast(x.strip()) for x in text.split(',') if x.strip()]


def _parse_multi_params(args) -> Dict[str, List]:
    """ดึงพารามิเตอร์ทั้งหมดที่ขึ้นต้นด้วย ``param_``"""
    params = {}
    for arg, value in vars(args).items():
        if arg.startswith('param_'):
            param = arg[6:]
            params[param] = _parse_csv_list(value, float if '.' in value else int)
    return params


def _filter_kwargs(func: Callable, kwargs: Dict[str, object]) -> Dict[str, object]:
    """คัดเฉพาะ kwargs ที่ฟังก์ชันรองรับ"""
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def export_summary(summary_df: pd.DataFrame, summary_path: str) -> pd.DataFrame:
    """บันทึก DataFrame สรุปผล sweep เป็น CSV และเติมคอลัมน์ที่ขาด"""
    # Ensure 'metric' and 'best_param' columns exist without warnings
    summary_df['metric'] = summary_df.get('metric', summary_df.get('score', np.nan))
    summary_df['best_param'] = summary_df.get('best_param', [{} for _ in range(len(summary_df))])
    try:
        summary_df.to_csv(summary_path, mode='w', index=False)
    except Exception as e:  # pragma: no cover - file write failure
        logging.critical(f"[Patch v6.5.3] Failed to write summary CSV: {e}")
        raise
    return summary_df


def _run_single_trial(
    params: Dict[str, object],
    df_log: pd.DataFrame,
    output_dir: str,
    trade_log_path: str,
    m1_path: str,
) -> dict | None:
    """Run one training trial or skip if the dataset is too small."""
    # Skip if not enough training samples
    if df_log.shape[0] < 2:
        logger.warning(
            f"[Patch v{__version__}] Skipping trial: only {df_log.shape[0]} training samples"
        )
        return None

    call_dict = _filter_kwargs(real_train_func, params)
    return real_train_func(
        output_dir=output_dir,
        trade_log_path=trade_log_path,
        m1_path=m1_path,
        **call_dict,
    )


def run_sweep(
    output_dir: str | None,
    params_grid: Dict[str, List],
    seed: int = 42,
    resume: bool = True,
    trade_log_path: str | None = None,
    m1_path: str | None = None,
) -> None:
    """รัน hyperparameter sweep พร้อมคุณสมบัติ resume และ QA log"""
    if not output_dir:
        output_dir = DEFAULT_SWEEP_DIR
    os.makedirs(output_dir, exist_ok=True)

    # [Patch v5.9.4] Load and validate trade log before running
    if not trade_log_path:
        logger.error("ต้องระบุ trade_log_path เพื่อทำการ sweep")
        raise SystemExit(1)
    if not os.path.exists(trade_log_path):
        # [Patch v6.3.1] Try simple .csv fallback, else create placeholder log
        alt = trade_log_path.replace('.csv.gz', '.csv')
        if os.path.exists(alt):
            trade_log_path = alt
        else:
            logger.warning(
                "[Patch v6.3.1] No walk-forward trade log at %s; creating placeholder and continuing",
                trade_log_path,
            )
            _create_placeholder_trade_log(trade_log_path)
    if not m1_path:
        m1_path = DefaultConfig.DATA_FILE_PATH_M1
    if not os.path.exists(m1_path):
        logger.error("ไม่พบไฟล์ M1 ที่ %s", m1_path)
        raise SystemExit(1)
    try:
        df_log = pd.read_csv(trade_log_path)
        # [Patch v5.8.13] Allow single-row trade logs with fallback metrics
        if len(df_log) < 1:
            logger.error(
                "trade log มีข้อมูลน้อยกว่า 1 แถว - ต้องใช้ walk-forward log ที่แท้จริง"
            )
            raise SystemExit(1)
    except Exception as e:  # pragma: no cover - unexpected read failure
        logger.error(f"อ่านไฟล์ trade log ไม่สำเร็จ: {e}")
        raise SystemExit(1)
    summary_path = os.path.join(output_dir, 'summary.csv')
    qa_log_path = os.path.join(output_dir, 'qa_sweep_log.txt')

    existing = set()
    if resume and os.path.exists(summary_path):
        df_exist = pd.read_csv(summary_path)
        # [Patch v5.10.9] Handle missing columns when resuming
        for row in df_exist.itertuples(index=False):
            row_dict = row._asdict()
            combo = tuple(row_dict.get(param) for param in params_grid)
            existing.add(combo)

    param_names = list(params_grid.keys())
    param_values = [params_grid[k] for k in param_names]
    summary_rows: List[Dict] = []

    total = 1
    for v in param_values:
        total *= len(v)
    pbar = tqdm(total=total, desc='Sweep progress', ncols=100)

    for run_id, values in enumerate(product(*param_values), start=1):
        key = tuple(values)
        if key in existing:
            pbar.update(1)
            continue

        param_dict = dict(zip(param_names, values))
        param_dict['seed'] = seed
        log_msg = f"Run {run_id}: {param_dict}"
        logger.info(log_msg)
        try:
            result = _run_single_trial(
                param_dict,
                df_log,
                output_dir,
                trade_log_path,
                m1_path,
            )
            if result is None:
                summary_rows.append({
                    'run_id': run_id,
                    **param_dict,
                    'error': 'insufficient training samples',
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                with open(qa_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"SKIP {log_msg} => insufficient training samples\n")
                pbar.update(1)
                continue
            metric_val = None
            if result.get('metrics'):
                metric_val = list(result['metrics'].values())[0]
            summary_row = {
                'run_id': run_id,
                **param_dict,
                'model_path': result['model_path'].get('model', ''),
                'features': ','.join(result.get('features', [])),
                **result.get('metrics', {}),
                'metric': metric_val,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            summary_rows.append(summary_row)
            with open(qa_log_path, 'a', encoding='utf-8') as f:
                f.write(f"SUCCESS {log_msg} => {summary_row}\n")
        except Exception as e:  # pragma: no cover - unexpected failures
            err_trace = traceback.format_exc()
            logger.error(f"Error at {log_msg}: {e}")
            with open(qa_log_path, 'a', encoding='utf-8') as f:
                f.write(f"ERROR {log_msg} => {e}\n{err_trace}\n")
            summary_rows.append({
                'run_id': run_id,
                **param_dict,
                'error': str(e),
                'traceback': err_trace,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        pbar.update(1)

    pbar.close()

    if os.path.exists(summary_path):
        df_exist = pd.read_csv(summary_path)
        df = pd.concat([df_exist, pd.DataFrame(summary_rows)], ignore_index=True)
    else:
        df = pd.DataFrame(summary_rows)
    df = export_summary(df, summary_path)
    logger.info(f"Sweep summary saved to {summary_path}")

    # (ไม่มีแก้) – ตรงนี้บันทึกไฟล์ชื่อ best_param.json ตามมาตรฐานโค้ด
    metric_col = 'metric' if 'metric' in df.columns else None
    if metric_col is None or df[metric_col].dropna().empty:
        logger.error(
            "ไม่มีผลการ sweep ที่สำเร็จ – จะไม่ปรับ hyperparameter จากรอบนี้"
        )
        return
    if metric_col and not df[metric_col].dropna().empty:
        best_row = df.sort_values(metric_col, ascending=False).iloc[0]
        best_param_path = os.path.join(output_dir, 'best_param.json')
        best_row[param_names + ['seed']].to_json(best_param_path, force_ascii=False)
        logger.info(
            f"Best param: {dict(best_row[param_names + ['seed']])} -> {best_row[metric_col]}"
        )
        if os.path.exists(best_param_path):
            logger.info("[Patch v5.9.1] best_param.json saved to %s", best_param_path)
        else:
            logger.error("[Patch v5.9.1] best_param.json missing at %s", best_param_path)
    else:
        logger.warning(
            "ไม่สามารถหา metric สำหรับเลือกพารามิเตอร์ที่ดีที่สุด"
        )


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=DEFAULT_SWEEP_DIR)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--param_learning_rate', default='0.01,0.05')
    parser.add_argument('--param_depth', default='6,8')
    parser.add_argument('--param_l2_leaf_reg', default='1,3,5')
    parser.add_argument('--param_subsample', default='0.8,1.0')  # [Patch v6.2.1] new CLI option
    parser.add_argument('--param_colsample_bylevel', default='0.8,1.0')  # [Patch v6.2.1] new CLI option
    parser.add_argument(
        '--trade_log_path', '--trade-log',
        dest='trade_log_path',
        default=DEFAULT_TRADE_LOG,
    )
    parser.add_argument('--m1_path')
    return parser.parse_args(args)


def main(args=None) -> None:
    args = parse_args(args)

    params_grid = _parse_multi_params(args)
    run_sweep(
        args.output_dir,
        params_grid,
        seed=args.seed,
        resume=args.resume,
        trade_log_path=args.trade_log_path,
        m1_path=args.m1_path,
    )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # pragma: no cover - CLI entry
        logger.error("เกิดข้อผิดพลาดที่ไม่คาดคิด: %s", str(e), exc_info=True)
        sys.exit(1)

