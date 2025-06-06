import os
import base64
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_dashboard(equity: pd.Series, drawdown: pd.Series, returns: pd.Series, shap_path: str | None = None) -> go.Figure:
    """Create interactive dashboard for walk-forward results."""
    if equity is None or equity.empty:
        raise ValueError("equity series is empty")
    if drawdown is None or drawdown.empty:
        raise ValueError("drawdown series is empty")
    if returns is None or returns.empty:
        raise ValueError("returns series is empty")

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Equity Curve", "Drawdown Curve", "Returns Histogram", "SHAP Summary"])
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity"), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown"), row=1, col=2)
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=50, name="Returns"), row=2, col=1)

    if shap_path and os.path.exists(shap_path):
        with open(shap_path, "rb") as f:
            enc = base64.b64encode(f.read()).decode("ascii")
        fig.add_layout_image(dict(source=f"data:image/png;base64,{enc}", xref="paper", yref="paper", x=1, y=0.4, sizex=0.9, sizey=0.5))

    fig.update_layout(height=800, width=1000, title_text="WFV Dashboard")
    return fig


def save_dashboard(fig: go.Figure, output_path: str) -> None:
    """Save plotly dashboard to HTML."""
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_wfv_summary(results: pd.DataFrame) -> go.Figure:
    """Create bar chart summarizing PnL per fold."""
    if results is None or results.empty:
        raise ValueError("results dataframe is empty")
    fig = go.Figure()
    fig.add_bar(x=results["fold"], y=results["test_pnl"], name="Test PnL")
    if "train_pnl" in results.columns:
        fig.add_bar(x=results["fold"], y=results["train_pnl"], name="Train PnL")
    fig.update_layout(
        barmode="group",
        title="WFV PnL per Fold",
        xaxis_title="Fold",
        yaxis_title="PnL",
        height=500,
        width=700,
    )
    return fig
