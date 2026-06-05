import pandas as pd
from datetime import datetime


def format_percent(value):
    try:
        return f"{value:.2%}"
    except Exception:
        return value


def format_number(value):
    try:
        return f"{value:.2f}"
    except Exception:
        return value


def generate_report():
    train_df = pd.read_csv("reports/train_results.csv")
    test_df = pd.read_csv("reports/test_results.csv")

    best_test = test_df.sort_values(by="Score", ascending=False).iloc[0]

    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""
    <html>
    <head>
        <title>AlphaLab Strategy Research Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #222;
            }}
            h1, h2, h3 {{
                color: #111;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
                margin-bottom: 30px;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: right;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            td:first-child, th:first-child {{
                text-align: left;
            }}
            .metric-box {{
                background-color: #f8f8f8;
                border-left: 5px solid #333;
                padding: 15px;
                margin: 20px 0;
            }}
            .warning {{
                background-color: #fff3cd;
                border-left: 5px solid #e0a800;
                padding: 15px;
                margin: 20px 0;
            }}
            .good {{
                background-color: #e8f5e9;
                border-left: 5px solid #2e7d32;
                padding: 15px;
                margin: 20px 0;
            }}
            code {{
                background-color: #eee;
                padding: 2px 5px;
                border-radius: 4px;
            }}
        </style>
    </head>

    <body>
        <h1>AlphaLab: Quantitative Strategy Research Report</h1>
        <p><strong>Generated:</strong> {report_date}</p>

        <h2>Project Summary</h2>
        <p>
            AlphaLab is a Python-based quantitative research project designed to test
            mean-reversion trading signals on historical equity data. The system downloads
            market data, engineers statistical features, runs backtests, applies transaction
            costs, performs train/test validation, and ranks strategies using risk-adjusted
            metrics.
        </p>

        <h2>Strategy Tested</h2>
        <p>
            The tested strategy is a <strong>mean-reversion strategy</strong>. It buys when
            a stock's price falls significantly below its moving average, measured by a
            rolling z-score, and exits when the price reverts back toward the average.
        </p>

        <p>
            Signal formula:
        </p>

        <p>
            <code>Z-Score = (Close Price - Rolling Moving Average) / Rolling Standard Deviation</code>
        </p>

        <h2>Validation Method</h2>
        <p>
            Each ticker was split into a training period and an out-of-sample testing period.
            Strategy parameters were selected using only the training data, then evaluated
            on unseen test data.
        </p>

        <ul>
            <li>Training set: first 70% of the data</li>
            <li>Test set: final 30% of the data</li>
            <li>Transaction cost: 0.05% per trade event</li>
            <li>Ranking metric: custom robust score using Sharpe, Sortino, profit factor, outperformance, drawdown, exposure, and trade count</li>
        </ul>

        <h2>Best Out-of-Sample Strategy</h2>

        <div class="metric-box">
            <h3>{best_test["Ticker"]} Mean-Reversion Strategy</h3>
            <p><strong>Entry Z:</strong> {best_test["Entry Z"]}</p>
            <p><strong>Window:</strong> {int(best_test["Window"])}</p>
            <p><strong>Strategy Return:</strong> {format_percent(best_test["Strategy Return"])}</p>
            <p><strong>Buy-and-Hold Return:</strong> {format_percent(best_test["Buy Hold Return"])}</p>
            <p><strong>Outperformance:</strong> {format_percent(best_test["Outperformance"])}</p>
            <p><strong>Sharpe Ratio:</strong> {format_number(best_test["Sharpe"])}</p>
            <p><strong>Sortino Ratio:</strong> {format_number(best_test["Sortino"])}</p>
            <p><strong>Profit Factor:</strong> {format_number(best_test["Profit Factor"])}</p>
            <p><strong>Max Drawdown:</strong> {format_percent(best_test["Max Drawdown"])}</p>
            <p><strong>Trades:</strong> {int(best_test["Trades"])}</p>
            <p><strong>Win Rate:</strong> {format_percent(best_test["Win Rate"])}</p>
            <p><strong>Exposure Time:</strong> {format_percent(best_test["Exposure Time"])}</p>
        </div>

        <div class="good">
            <strong>Main Finding:</strong>
            The best out-of-sample result came from {best_test["Ticker"]}, where the strategy
            outperformed buy-and-hold during the test period.
        </div>

        <div class="warning">
            <strong>Important Limitation:</strong>
            The strategy still needs stronger risk controls. A high return does not automatically
            mean the strategy is reliable, especially when max drawdown is large or the number of
            trades is limited.
        </div>

        <h2>Out-of-Sample Test Results</h2>
        {test_df.to_html(index=False)}

        <h2>Top Training Results</h2>
        {train_df.head(10).to_html(index=False)}

        <h2>Interpretation</h2>
        <p>
            The results suggest that the mean-reversion signal performs better on volatile
            individual stocks than on broad market ETFs. TSLA produced the strongest out-of-sample
            result, while SPY and QQQ were weaker. This suggests the strategy is not universal and
            may depend heavily on volatility, ticker behavior, and market regime.
        </p>

        <h2>Next Research Improvements</h2>
        <ul>
            <li>Add stop-loss and take-profit rules</li>
            <li>Add volatility-based position sizing</li>
            <li>Test more tickers and longer time periods</li>
            <li>Add walk-forward validation</li>
            <li>Compare against random-entry baseline strategies</li>
            <li>Test whether results remain after higher transaction costs and slippage</li>
        </ul>

        <h2>Conclusion</h2>
        <p>
            AlphaLab demonstrates the full quant research workflow: hypothesis generation,
            data collection, feature engineering, backtesting, train/test validation, performance
            measurement, and honest evaluation of limitations.
        </p>
    </body>
    </html>
    """

    with open("reports/strategy_report.html", "w") as file:
        file.write(html)

    print("Report created: reports/strategy_report.html")


if __name__ == "__main__":
    generate_report()
