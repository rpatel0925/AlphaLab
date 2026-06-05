import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AlphaLab Quant Research Dashboard",
    layout="wide"
)

st.title("AlphaLab: Quantitative Strategy Research Dashboard")

st.write(
    """
    AlphaLab tests mean-reversion trading strategies using historical market data,
    transaction costs, train/test validation, and risk-adjusted performance metrics.
    """
)

@st.cache_data
def load_data():
    train_df = pd.read_csv("reports/train_results.csv")
    test_df = pd.read_csv("reports/test_results.csv")
    return train_df, test_df

train_df, test_df = load_data()

st.header("Best Out-of-Sample Strategy")

best = test_df.sort_values(by="Score", ascending=False).iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", best["Ticker"])
col2.metric("Strategy Return", f"{best['Strategy Return']:.2%}")
col3.metric("Buy & Hold Return", f"{best['Buy Hold Return']:.2%}")
col4.metric("Outperformance", f"{best['Outperformance']:.2%}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Sharpe", f"{best['Sharpe']:.2f}")
col6.metric("Sortino", f"{best['Sortino']:.2f}")
col7.metric("Max Drawdown", f"{best['Max Drawdown']:.2%}")
col8.metric("Win Rate", f"{best['Win Rate']:.2%}")

st.header("Out-of-Sample Test Results")
st.dataframe(test_df, use_container_width=True)

st.header("Filter Results")

tickers = sorted(test_df["Ticker"].unique())
selected_tickers = st.multiselect("Select tickers", tickers, default=tickers)

filtered_df = test_df[test_df["Ticker"].isin(selected_tickers)]
st.dataframe(filtered_df, use_container_width=True)

st.header("Top Training Results")
st.dataframe(train_df.head(20), use_container_width=True)

st.header("Research Notes")

st.markdown(
    """
    **Main takeaway:**  
    The strategy performed best on volatile individual stocks, especially TSLA.

    **Important limitation:**  
    High returns do not automatically mean the strategy is reliable. Some results still have
    high drawdowns and limited trade counts.

    **Next improvements:**  
    Add stop-loss rules, take-profit rules, volatility-based position sizing, walk-forward
    validation, and random-entry baseline testing.
    """
)
