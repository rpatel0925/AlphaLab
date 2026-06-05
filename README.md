# AlphaLab: Quantitative Strategy Research Engine
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alphalab-bpencwgdf4nneibfxfrwhn.streamlit.app/)

Live dashboard: https://alphalab-bpencwgdf4nneibfxfrwhn.streamlit.app/

AlphaLab is a Python-based quantitative research project that tests mean-reversion trading strategies on historical equity data.

The project demonstrates a full quant research workflow:

- Data collection
- Feature engineering
- Backtesting
- Transaction cost modeling
- Train/test validation
- Risk-adjusted performance analysis
- Strategy reporting
- Interactive dashboard

## Strategy Overview

The strategy tested is a mean-reversion strategy.

It buys when a stock's price falls significantly below its rolling moving average, measured using a z-score.

```text
Z-Score = (Close Price - Rolling Moving Average) / Rolling Standard Deviation
