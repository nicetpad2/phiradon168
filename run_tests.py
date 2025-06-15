import argparse
import logging
import os
import sys
import subprocess
import pytest

# [Patch v6.9.57] Add --keyword option to filter tests by expression
# [Patch v6.9.56] Add --cov-fail-under option for coverage threshold

# [Patch v6.9.52] Add coverage and maxfail options, auto maxfail with --fast
# [Patch v6.9.54] Auto-select changed tests with --fast and add --durations option

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


def find_changed_tests(base_ref: str) -> list[str]:
    """Return test files changed since base_ref."""
    try:
        output = subprocess.check_output(
            ['git', 'diff', '--name-only', f'{base_ref}..HEAD'],
            text=True,
        )
    except Exception:
        return []
    return [f.strip() for f in output.splitlines() if f.startswith('tests/') and f.endswith('.py')]


def main() -> None:
    parser = argparse.ArgumentParser(description='รัน test suite')
    parser.add_argument('--fast', '--smoke', action='store_true', dest='fast',
                        help='ข้าม integration tests ที่ใช้เวลานาน และรันเฉพาะไฟล์ที่แก้ไข')
    parser.add_argument('-n', '--num-processes', default=None,
                        help='จำนวน process สำหรับรันแบบขนาน (pytest-xdist)')
    parser.add_argument('--lf', '--last-failed', action='store_true', dest='last_failed',
                        help='รันเฉพาะเทสที่ล้มเหลวครั้งก่อน')
    parser.add_argument('--cov', nargs='?', const='src', metavar='TARGET', default=None,
                        help='วัด coverage ของ TARGET (ค่าเริ่มต้น src)')
    parser.add_argument('--maxfail', type=int, default=None, metavar='N',
                        help='หยุดทันทีเมื่อมี N เทสล้มเหลว')
    parser.add_argument('--cov-fail-under', type=int, dest='cov_fail_under',
                        default=None, metavar='PERCENT',
                        help='ล้มเหลวหาก coverage ต่ำกว่า PERCENT')
    parser.add_argument('--durations', type=int, default=None, metavar='N',
                        help='แสดงรายการเทสที่ช้าที่สุด N อันดับ')
    parser.add_argument('-k', '--keyword', dest='keyword', default=None,
                        metavar='EXPR',
                        help='รันเฉพาะเทสที่ตรงกับ EXPR')
    parser.add_argument('-c', '--changed', nargs='?', const='HEAD~1', default=None,
                        metavar='BASE',
                        help='รันเฉพาะเทสที่เปลี่ยนจาก BASE (ค่าเริ่มต้น HEAD~1)')
    args, extra_args = parser.parse_known_args()

    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.WARNING)

    pytest_args = extra_args
    if not pytest_args:
        base = args.changed
        if args.fast and base is None:
            base = 'HEAD~1'
        if base:
            pytest_args = find_changed_tests(base)
            if pytest_args:
                print(f'[INFO] Running changed tests since {base}')
        if not pytest_args:
            pytest_args = ['tests']
    pytest_args.insert(0, '-q')
    if args.fast:
        pytest_args += ['-m', 'not integration']
        if args.maxfail is None:
            args.maxfail = 1

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

    if args.cov is not None:
        pytest_args += ['--cov', args.cov, '--cov-report', 'term-missing']
        if args.cov_fail_under is not None:
            pytest_args += ['--cov-fail-under', str(args.cov_fail_under)]
    elif args.cov_fail_under is not None:
        pytest_args += ['--cov', 'src', '--cov-report', 'term-missing',
                        '--cov-fail-under', str(args.cov_fail_under)]

    if args.maxfail is not None:
        pytest_args += ['--maxfail', str(args.maxfail)]

    if args.durations is not None:
        pytest_args += ['--durations', str(args.durations)]

    if args.keyword:
        pytest_args += ['-k', args.keyword]

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
