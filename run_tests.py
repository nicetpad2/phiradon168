import argparse
import logging
import os
import pytest

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


def main() -> None:
    parser = argparse.ArgumentParser(description='รัน test suite')
    parser.add_argument('--fast', '--smoke', action='store_true', dest='fast',
                        help='ข้าม integration tests ที่ใช้เวลานาน')
    parser.add_argument('-n', '--num-processes', default=None,
                        help='จำนวน process สำหรับรันแบบขนาน (pytest-xdist)')
    args = parser.parse_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.WARNING)

    pytest_args = [
        '--cov=.',
        '--cov-config=.coveragerc',
        '--cov-report=term',
        '-q',
    ]
    if args.fast:
        pytest_args += ['-m', 'not integration']

    if args.num_processes:
        try:
            import xdist  # noqa: F401
            pytest_args += ['-n', str(args.num_processes)]
        except ImportError:
            print('[WARN] pytest-xdist ไม่ได้ติดตั้ง จึงรันแบบปกติ')

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
