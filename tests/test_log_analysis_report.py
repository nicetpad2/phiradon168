from src.log_analysis import plot_trade_log_metrics

def test_plot_trade_log_metrics(tmp_path):
    log_file = tmp_path / 'log.txt'
    log_file.write_text(
        'INFO:root:   Attempting to Open New Order for BUY at 2023-01-01 10:00:00+00:00\n'
        'INFO:root:      Order Closing: Time=2023-01-01 10:30:00+00:00, Final Reason=TP, ExitPrice=1900, EntryTime=2023-01-01 10:00:00+00:00\n'
        'INFO:root:         [Patch PnL Final] Closed Lot=0.01, PnL(Net USD)=1.0\n'
    )
    fig = plot_trade_log_metrics(str(log_file))
    assert hasattr(fig, 'savefig')
