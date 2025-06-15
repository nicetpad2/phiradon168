# [Patch v6.9.39] Extracted strategy components from strategy.py
"""Basic strategy classes separated from :mod:`src.strategy`."""

# ---------------------------------------------------------------------------
# Strategy Pattern Utilities
# ---------------------------------------------------------------------------


class EntryStrategy:
    """Base class for entry signal generation."""

    def check_entry(self, data):
        raise NotImplementedError


class ExitStrategy:
    """Base class for exit signal generation."""

    def check_exit(self, data):
        raise NotImplementedError


class RiskManagementStrategy:
    """Base class for risk management adjustments."""

    def manage_risk(self, signal, data):
        return signal


class TrendFilter:
    """Base class for validating market trend."""

    def is_valid(self, data) -> bool:
        return True


class DefaultEntryStrategy(EntryStrategy):
    """Default entry handler using ``generate_open_signals``."""

    def check_entry(self, data):
        from strategy.entry_rules import generate_open_signals

        signals = generate_open_signals(data)
        return int(signals[-1]) if len(signals) else 0


class DefaultExitStrategy(ExitStrategy):
    """Default exit handler using ``generate_close_signals``."""

    def check_exit(self, data):
        from strategy.exit_rules import generate_close_signals

        signals = generate_close_signals(data)
        return int(signals[-1]) if len(signals) else 0


class MainStrategy:
    """Compose entry/exit/risk/trend components via dependency injection."""

    def __init__(self, entry_handler, exit_handler, risk_manager=None, trend_filter=None):
        self.entry_handler = entry_handler
        self.exit_handler = exit_handler
        self.risk_manager = risk_manager or RiskManagementStrategy()
        self.trend_filter = trend_filter or TrendFilter()

    def get_signal(self, data):
        if not self.trend_filter.is_valid(data):
            return 0
        signal = self.entry_handler.check_entry(data)
        return self.risk_manager.manage_risk(signal, data)


__all__ = [
    "EntryStrategy",
    "ExitStrategy",
    "RiskManagementStrategy",
    "TrendFilter",
    "DefaultEntryStrategy",
    "DefaultExitStrategy",
    "MainStrategy",
]
