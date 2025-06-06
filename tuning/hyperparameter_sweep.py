# -*- coding: utf-8 -*-
"""
[Patch v5.3.0] Hyperparameter sweep (Enterprise Edition)
- รองรับ multi-param sweep
- Save log + summary + best param
- Resume ได้ (skip run ที่เสร็จ)
- สรุปสถิติ + best config
"""
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

from nicegold.config import logger, DefaultConfig
from nicegold.training import real_train_func


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


def run_sweep(
    output_dir: str,
    params_grid: Dict[str, List],
    seed: int = 42,
    resume: bool = True,
    trade_log_path: str | None = None,
    m1_path: str | None = None,
) -> None:
    """รัน hyperparameter sweep พร้อมคุณสมบัติ resume และ QA log"""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'summary.csv')
    qa_log_path = os.path.join(output_dir, 'qa_sweep_log.txt')

    existing = set()
    if resume and os.path.exists(summary_path):
        df_exist = pd.read_csv(summary_path)
        existing = set(
            tuple(getattr(row, param) for param in params_grid)
            for row in df_exist.itertuples(index=False)
        )

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
            call_dict = _filter_kwargs(real_train_func, param_dict)
            result = real_train_func(
                output_dir=output_dir,
                trade_log_path=trade_log_path,
                m1_path=m1_path or DefaultConfig.DATA_FILE_PATH_M1,
                **call_dict,
            )
            summary_row = {
                'run_id': run_id,
                **param_dict,
                'model_path': result['model_path'].get('model', ''),
                'features': ','.join(result.get('features', [])),
                **result.get('metrics', {}),
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
    df.to_csv(summary_path, index=False)
    logger.info(f"Sweep summary saved to {summary_path}")

    metric_candidates = ['score', 'accuracy', 'f1', 'auc']
    metric_used = next((m for m in metric_candidates if m in df.columns), None)
    if metric_used:
        best_row = df.sort_values(metric_used, ascending=False).iloc[0]
        best_param_path = os.path.join(output_dir, 'best_param.json')
        best_row[param_names + ['seed']].to_json(best_param_path, force_ascii=False)
        logger.info(
            f"Best param ({metric_used}): {dict(best_row[param_names + ['seed']])} -> {best_row[metric_used]}"
        )
    else:
        logger.warning("No metric column found for best_param export.")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='sweep_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--param_learning_rate', default='0.01,0.05')
    parser.add_argument('--param_depth', default='6,8')
    parser.add_argument('--param_l2_leaf_reg', default='1,3,5')
    parser.add_argument(
        '--trade_log_path',
        default='./output_default/trade_log_v32_walkforward.csv.gz',
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

