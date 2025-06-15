import pandas as pd
from src.strategy_components import (
    EntryStrategy,
    ExitStrategy,
    RiskManagementStrategy,
    TrendFilter,
    DefaultEntryStrategy,
    DefaultExitStrategy,
    MainStrategy,
)

class DummyEntry(DefaultEntryStrategy):
    def check_entry(self, data):
        return 1

class DummyExit(DefaultExitStrategy):
    def check_exit(self, data):
        return 0

def test_imports_available():
    assert issubclass(DefaultEntryStrategy, EntryStrategy)
    assert issubclass(DefaultExitStrategy, ExitStrategy)


def test_main_strategy_integration():
    st = MainStrategy(DummyEntry(), DummyExit())
    assert st.get_signal(pd.DataFrame()) == 1
