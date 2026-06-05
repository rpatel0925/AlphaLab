import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def download_data(ticker="SPY", period="5y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data


def add_features(data, window=20):
    df = data.copy()

    df["Return"] = df["Close"].pct_change()
    df["Moving_Avg"] = df["Close"].rolling(window).mean()
    df["Rolling_Std"] = df["Close"].rolling(window).std()
    df["Z_Score"] = (df["Close"] - df["Moving_Avg"]) / df["Rolling_Std"]

    return df.dropna()


def run_strategy(df, entry_z=-1.5, exit_z=0.0, transaction_cost=0.0005):
    df = df.copy()
    df["Position"] = 0

    in_trade = False

    for i in range(len(df)):
        z = float(df["Z_Score"].iloc[i])

        if not in_trade and z < entry_z:
            in_trade = True
            df.iloc[i, df.columns.get_loc("Position")] = 1

        elif in_trade and z >= exit_z:
            in_trade = False
            df.iloc[i, df.columns.get_loc("Position")] = 0

        elif in_trade:
            df.iloc[i, df.columns.get_loc("Position")] = 1

    df["Raw_Strategy_Return"] = df["Position"].shift(1) * df["Return"]

    df["Trade"] = df["Position"].diff().abs().fillna(0)
    df["Cost"] = df["Trade"] * transaction_cost
    df["Strategy_Return"] = df["Raw_Strategy_Return"] - df["Cost"]

    return df.dropna()


def calculate_profit_factor(strategy_returns):
    gains = strategy_returns[strategy_returns > 0].sum()
    losses = strategy_returns[strategy_returns < 0].sum()

    if losses == 0:
        if gains > 0:
            return np.inf
        return 0

    return gains / abs(losses)


def calculate_sortino(strategy_returns):
    downside_returns = strategy_returns[strategy_returns < 0]

    if downside_returns.std() == 0 or np.isnan(downside_returns.std()):
        return 0

    return np.sqrt(252) * strategy_returns.mean() / downside_returns.std()


def calculate_metrics(df):
    if df.empty:
        return None, None

    strategy_return = (1 + df["Strategy_Return"]).prod() - 1
    buy_hold_return = (1 + df["Return"]).prod() - 1
    outperformance = strategy_return - buy_hold_return

    if df["Strategy_Return"].std() == 0:
        sharpe = 0
    else:
        sharpe = np.sqrt(252) * df["Strategy_Return"].mean() / df["Strategy_Return"].std()

    sortino = calculate_sortino(df["Strategy_Return"])
    profit_factor = calculate_profit_factor(df["Strategy_Return"])

    equity_curve = (1 + df["Strategy_Return"]).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    trades = int((df["Position"].diff() == 1).sum())
    position_changes = df["Position"].diff().fillna(0)

    exposure_time = df["Position"].mean()

    if exposure_time > 0:
        return_per_exposure = strategy_return / exposure_time
    else:
        return_per_exposure = 0

    win_trades = 0
    losing_trades = 0
    trade_returns = []
    entry_price = None

    for i in range(len(df)):
        pos_change = position_changes.iloc[i]

        if pos_change == 1:
            entry_price = float(df["Close"].iloc[i])

        elif pos_change == -1 and entry_price is not None:
            exit_price = float(df["Close"].iloc[i])
            trade_return = (exit_price / entry_price) - 1
            trade_returns.append(trade_return)

            if trade_return > 0:
                win_trades += 1
            else:
                losing_trades += 1

            entry_price = None

    total_closed_trades = win_trades + losing_trades

    if total_closed_trades > 0:
        win_rate = win_trades / total_closed_trades
        avg_trade_return = np.mean(trade_returns)
    else:
        win_rate = 0
        avg_trade_return = 0

    return {
        "Strategy Return": strategy_return,
        "Buy Hold Return": buy_hold_return,
        "Outperformance": outperformance,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_drawdown,
        "Trades": trades,
        "Closed Trades": total_closed_trades,
        "Win Rate": win_rate,
        "Avg Trade Return": avg_trade_return,
        "Exposure Time": exposure_time,
        "Return Per Exposure": return_per_exposure,
    }, equity_curve


def evaluate_strategy(data, ticker, entry_z, window):
    df = add_features(data, window=window)
    results = run_strategy(df, entry_z=entry_z)
    metrics, equity_curve = calculate_metrics(results)

    if metrics is None:
        return None, None, None

    row = {
        "Ticker": ticker,
        "Entry Z": entry_z,
        "Window": window,
        **metrics,
    }

    return row, results, equity_curve


def split_train_test(data, train_size=0.7):
    split_index = int(len(data) * train_size)
    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()
    return train, test


def strategy_score(row, min_trades=10):
    """
    Honest ranking formula.

    We reward:
    - Positive Sharpe
    - Positive outperformance
    - More trades
    - Lower drawdown

    We punish:
    - Too few trades
    - Negative returns
    - High drawdowns
    """

    if row["Trades"] < min_trades:
        return -999

    if row["Strategy Return"] <= 0:
        return -999

    score = 0
    score += row["Sharpe"] * 2
    score += row["Sortino"]
    score += row["Outperformance"] * 2
    score += row["Return Per Exposure"]
    score += row["Profit Factor"] * 0.25 if np.isfinite(row["Profit Factor"]) else 1
    score += row["Max Drawdown"]

    return score


def run_train_test_research():
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]
    entry_levels = [-1.0, -1.5, -2.0, -2.5]
    windows = [10, 20, 50]
    min_trades = 10

    train_rows = []
    test_rows = []

    print("Running robust train/test strategy research...\n")

    for ticker in tickers:
        try:
            data = download_data(ticker=ticker)
            train_data, test_data = split_train_test(data, train_size=0.7)

            best_train_row = None
            best_params = None
            best_train_score = -999

            for entry_z in entry_levels:
                for window in windows:
                    row, results, equity_curve = evaluate_strategy(
                        data=train_data,
                        ticker=ticker,
                        entry_z=entry_z,
                        window=window
                    )

                    if row is None:
                        continue

                    row["Score"] = strategy_score(row, min_trades=min_trades)
                    train_rows.append(row)

                    if row["Score"] > best_train_score:
                        best_train_score = row["Score"]
                        best_train_row = row
                        best_params = {
                            "entry_z": entry_z,
                            "window": window
                        }

            if best_params is None:
                print(f"No valid strategy found for {ticker}")
                continue

            print(
                f"Best training params for {ticker}: "
                f"entry_z={best_params['entry_z']}, "
                f"window={best_params['window']}, "
                f"score={best_train_score:.2f}"
            )

            test_row, test_results, test_equity_curve = evaluate_strategy(
                data=test_data,
                ticker=ticker,
                entry_z=best_params["entry_z"],
                window=best_params["window"]
            )

            if test_row is not None:
                test_row["Chosen From Training Entry Z"] = best_params["entry_z"]
                test_row["Chosen From Training Window"] = best_params["window"]
                test_row["Score"] = strategy_score(test_row, min_trades=3)
                test_rows.append(test_row)

        except Exception as e:
            print(f"Error with {ticker}: {e}")

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_df = train_df.sort_values(by="Score", ascending=False)
    test_df = test_df.sort_values(by="Score", ascending=False)

    train_df.to_csv("reports/train_results.csv", index=False)
    test_df.to_csv("reports/test_results.csv", index=False)

    print("\n--- Best Training Results Using Robust Score ---")
    print(train_df.head(10).to_string(index=False))

    print("\n--- Out-of-Sample Test Results ---")
    print(test_df.to_string(index=False))

    print("\nSaved files:")
    print("reports/train_results.csv")
    print("reports/test_results.csv")

    if not test_df.empty:
        best_test = test_df.iloc[0]

        ticker = best_test["Ticker"]
        entry_z = best_test["Entry Z"]
        window = int(best_test["Window"])

        full_data = download_data(ticker=ticker)
        train_data, test_data = split_train_test(full_data, train_size=0.7)

        _, test_results, test_equity_curve = evaluate_strategy(
            data=test_data,
            ticker=ticker,
            entry_z=entry_z,
            window=window
        )

        buy_hold = (1 + test_results["Return"]).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(test_equity_curve, label=f"{ticker} Strategy Out-of-Sample")
        plt.plot(buy_hold, label=f"{ticker} Buy and Hold Out-of-Sample")
        plt.title("Out-of-Sample Strategy vs Buy and Hold")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    run_train_test_research()
