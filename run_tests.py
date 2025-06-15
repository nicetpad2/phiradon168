import argparse
import logging
import os
import sys
import pytest

# [Patch v6.9.51] เพิ่มตัวเลือก coverage และ rerun เพื่อลดเวลาและตรวจสอบครบถ้วน

class _SummaryPlugin:
    """Plugin เก็บสถิติผลการทดสอบ"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.total += 1
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1
            elif report.skipped:
                self.skipped += 1


def _has_plugin(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description='รัน test suite')
    parser.add_argument('--fast', '--smoke', action='store_true', dest='fast',
                        help='ข้าม integration tests ที่ใช้เวลานาน')
    parser.add_argument('-n', '--num-processes', default=None,
                        help='จำนวน process สำหรับรันแบบขนาน (pytest-xdist)')
    parser.add_argument('--lf', '--last-failed', action='store_true', dest='last_failed',
                        help='รันเฉพาะเทสที่ล้มเหลวครั้งก่อน')
    parser.add_argument('--cov', action='store_true',
                        help='เปิดการวัด coverage หากมี pytest-cov')
    parser.add_argument('--no-cov', action='store_true',
                        help='ปิดการวัด coverage แม้มี pytest-cov')
    parser.add_argument('--reruns', type=int, default=0,
                        help='จำนวนครั้ง rerun เมื่อเทสล้มเหลว (pytest-rerunfailures)')
    args, extra_args = parser.parse_known_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.WARNING)

    pytest_args = extra_args
    if not pytest_args:
        pytest_args = ['tests']
    pytest_args.insert(0, '-q')
    if args.fast:
        pytest_args += ['-m', 'not integration']

    if args.num_processes:
        try:
            import xdist  # noqa: F401
            pytest_args += ['-n', str(args.num_processes)]
        except ImportError:
            print('[WARN] pytest-xdist ไม่ได้ติดตั้ง จึงรันแบบปกติ')
    else:
        try:
            import xdist  # noqa: F401
            pytest_args += ['-n', 'auto']
        except ImportError:
            pass

    if args.last_failed:
        pytest_args += ['--last-failed', '--last-failed-no-failures', 'all']

    if args.reruns and _has_plugin('pytest_rerunfailures'):
        pytest_args += ['--reruns', str(args.reruns)]

    use_cov = args.cov or (os.environ.get('PYTEST_COV') and not args.no_cov)
    if use_cov and _has_plugin('pytest_cov'):
        pytest_args += ['--cov=src', '--cov-report=term-missing']
        fail_under = os.environ.get('COV_FAIL_UNDER')
        if fail_under:
            pytest_args += ['--cov-fail-under', fail_under]

    summary = _SummaryPlugin()
    exit_code = pytest.main(pytest_args, plugins=[summary])

    if summary.total == 0:
        print('[SUMMARY] No tests collected')
        exit_code = exit_code or 1
    else:
        print(f"[SUMMARY] Total tests: {summary.total}, "
              f"Passed: {summary.passed}, Failed: {summary.failed}, "
              f"Skipped: {summary.skipped}")

    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
