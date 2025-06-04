import os
import logging
import pytest

if __name__ == '__main__':
    os.environ.setdefault('COMPACT_LOG', '1')
    logging.basicConfig(level=logging.WARNING)
    exit_code = pytest.main(['-q'])
    if exit_code == 0:
        print('[SUMMARY] All tests passed')
    else:
        print(f'[SUMMARY] Tests exited with code {exit_code}')
    raise SystemExit(exit_code)
