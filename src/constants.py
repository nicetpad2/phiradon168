class ColumnName:
    """Standard column names used throughout the project."""

    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'

    OPEN_CAP = 'Open'
    HIGH_CAP = 'High'
    LOW_CAP = 'Low'
    CLOSE_CAP = 'Close'
    VOLUME_CAP = 'Volume'


class Signal:
    """Integer representations of trade signals."""

    LONG = 1
    SHORT = -1
    NEUTRAL = 0

