"""Main namespace for the NiceGold project."""
from .adaptive import (
    adaptive_sl_tp,
    adaptive_risk,
    log_best_params,
    calculate_atr,
    atr_position_size,
)
from .money_management import (
    atr_sl_tp,
    update_be_trailing,
    adaptive_position_size,
    portfolio_hard_stop,
)
from .evaluation import evaluate_meta_classifier
from .wfv import walk_forward_grid_search, prune_features_by_importance

__all__ = [
    'adaptive_sl_tp',
    'adaptive_risk',
    'log_best_params',
    'calculate_atr',
    'atr_position_size',
    'atr_sl_tp',
    'update_be_trailing',
    'adaptive_position_size',
    'portfolio_hard_stop',
    'evaluate_meta_classifier',
    'walk_forward_grid_search',
    'prune_features_by_importance',
]
