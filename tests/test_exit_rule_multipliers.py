import pandas as pd
import numpy as np
import src.strategy as strategy

def test_precompute_arrays_custom_mult():
    df = pd.DataFrame({'Close': [1, 2, 3], 'ATR_14': [0.2, 0.2, 0.2]})
    sl_default = strategy.precompute_sl_array(df)
    tp_default = strategy.precompute_tp_array(df)
    assert np.isclose(sl_default[-1], 0.4)
    assert np.isclose(tp_default[-1], 0.4)
    sl_custom = strategy.precompute_sl_array(df, sl_mult=1.5)
    tp_custom = strategy.precompute_tp_array(df, tp_mult=3.0)
    assert np.isclose(sl_custom[-1], 0.3)
    assert np.isclose(tp_custom[-1], 0.6)
