import pandas as pd
import sys
sys.path.insert(0, '.')
from src.strategy import (
    MainStrategy,
    EntryStrategy,
    ExitStrategy,
    RiskManagementStrategy,
    TrendFilter,
)

class DummyEntry(EntryStrategy):
    def __init__(self, result):
        self.result = result
    def check_entry(self, data):
        return self.result

class DummyExit(ExitStrategy):
    def check_exit(self, data):
        return 0

class DummyRisk(RiskManagementStrategy):
    def manage_risk(self, signal, data):
        return signal * 2

class DummyFilter(TrendFilter):
    def __init__(self, valid):
        self.valid = valid
    def is_valid(self, data):
        return self.valid

def test_main_strategy_entry():
    st = MainStrategy(DummyEntry(1), DummyExit(), DummyRisk(), DummyFilter(True))
    assert st.get_signal(pd.DataFrame()) == 2

def test_main_strategy_filter_blocks():
    st = MainStrategy(DummyEntry(1), DummyExit(), DummyRisk(), DummyFilter(False))
    assert st.get_signal(pd.DataFrame()) == 0
