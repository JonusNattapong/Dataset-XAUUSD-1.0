#!/usr/bin/env python3
"""
XAUUSD Trading Strategy Backtesting Framework

This module provides a comprehensive backtesting framework for testing
trading strategies on the XAUUSD dataset with realistic trading simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    position_size: float
    direction: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    holding_period: int

@dataclass
class BacktestResult:
    """Results from a backtest"""
    trades: List[Trade]
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series

class Backtester:
    """Main backtesting engine for XAUUSD trading strategies"""

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize backtester

        Args:
            data: DataFrame with OHLCV and signals
            initial_capital: Starting capital for backtest
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0  # Current position size
        self.trades = []
        self.equity_curve = []

        # Validate data
        required_cols = ['Date', 'Close', 'Open', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
            self.data['Date'] = pd.to_datetime(self.data['Date'])

        self.data = self.data.sort_values('Date').reset_index(drop=True)

    def generate_signals(self, strategy_func: Callable) -> pd.DataFrame:
        """
        Generate trading signals using a strategy function

        Args:
            strategy_func: Function that takes DataFrame and returns signals

        Returns:
            DataFrame with signal columns added
        """
        return strategy_func(self.data.copy())

    def run_backtest(self, signals_df: pd.DataFrame,
                    position_size_pct: float = 0.1,
                    stop_loss_pct: float = 0.02,
                    take_profit_pct: float = 0.05,
                    max_holding_period: int = 20) -> BacktestResult:
        """
        Run the backtest with given signals

        Args:
            signals_df: DataFrame with 'signal' column (1=buy, -1=sell, 0=hold)
            position_size_pct: Position size as percentage of capital
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_holding_period: Maximum days to hold a position

        Returns:
            BacktestResult with performance metrics
        """

        df = signals_df.copy()
        df['signal'] = df.get('signal', 0)

        current_position = 0
        entry_price = 0
        entry_date = None
        equity = [self.initial_capital]

        for idx, row in df.iterrows():
            current_price = row['Close']
            signal = row['signal']
            current_date = row['Date']

            # Check for exit conditions if we have a position
            if current_position != 0:
                holding_period = (current_date - entry_date).days

                # Calculate P&L
                if current_position > 0:  # Long position
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # Short position
                    pnl_pct = (entry_price - current_price) / entry_price

                # Exit conditions
                exit_signal = False
                exit_reason = ""

                # Stop loss
                if abs(pnl_pct) >= stop_loss_pct:
                    exit_signal = True
                    exit_reason = "stop_loss"

                # Take profit
                elif pnl_pct >= take_profit_pct:
                    exit_signal = True
                    exit_reason = "take_profit"

                # Max holding period
                elif holding_period >= max_holding_period:
                    exit_signal = True
                    exit_reason = "max_holding"

                # Opposite signal
                elif (current_position > 0 and signal == -1) or (current_position < 0 and signal == 1):
                    exit_signal = True
                    exit_reason = "signal"

                # Execute exit
                if exit_signal:
                    position_value = abs(current_position) * entry_price
                    pnl = position_value * pnl_pct

                    self.current_capital += pnl

                    # Record trade
                    trade = Trade(
                        entry_date=str(entry_date.date()),
                        exit_date=str(current_date.date()),
                        entry_price=entry_price,
                        exit_price=current_price,
                        position_size=abs(current_position),
                        direction='long' if current_position > 0 else 'short',
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        holding_period=holding_period
                    )
                    self.trades.append(trade)

                    current_position = 0
                    entry_price = 0
                    entry_date = None

            # Check for entry signals if we don't have a position
            if current_position == 0 and signal != 0:
                # Calculate position size
                position_value = self.current_capital * position_size_pct
                position_size = position_value / current_price

                current_position = position_size if signal == 1 else -position_size
                entry_price = current_price
                entry_date = current_date

            # Update equity curve
            equity.append(self.current_capital)

        # Close any remaining position at the end
        if current_position != 0:
            final_price = df.iloc[-1]['Close']
            final_date = df.iloc[-1]['Date']
            holding_period = (final_date - entry_date).days

            if current_position > 0:
                pnl_pct = (final_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - final_price) / entry_price

            position_value = abs(current_position) * entry_price
            pnl = position_value * pnl_pct
            self.current_capital += pnl

            trade = Trade(
                entry_date=str(entry_date.date()),
                exit_date=str(final_date.date()),
                entry_price=entry_price,
                exit_price=final_price,
                position_size=abs(current_position),
                direction='long' if current_position > 0 else 'short',
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_period=holding_period
            )
            self.trades.append(trade)

        # Calculate performance metrics
        result = self._calculate_metrics(equity)

        return result

    def _calculate_metrics(self, equity: List[float]) -> BacktestResult:
        """Calculate performance metrics from equity curve"""

        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()

        # Basic metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / equity_series.iloc[0]

        # Annualized return (assuming daily data)
        days = len(equity_series) - 1
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0

        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()

        # Trade metrics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]

            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

            if avg_loss != 0:
                profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades]))
            else:
                profit_factor = float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return BacktestResult(
            trades=self.trades,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            equity_curve=equity_series
        )

def plot_backtest_results(result: BacktestResult, title: str = "Backtest Results"):
    """Plot backtest results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Equity curve
    axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
    axes[0, 0].set_title('Equity Curve')
    axes[0, 0].set_xlabel('Trade Number')
    axes[0, 0].set_ylabel('Portfolio Value')
    axes[0, 0].grid(True)

    # Drawdown
    peak = result.equity_curve.expanding().max()
    drawdown = (result.equity_curve - peak) / peak
    axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    axes[0, 1].set_title('Drawdown')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Drawdown %')
    axes[0, 1].grid(True)

    # Trade P&L distribution
    if result.trades:
        pnl_values = [t.pnl for t in result.trades]
        axes[1, 0].hist(pnl_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].axvline(np.mean(pnl_values), color='red', linestyle='--', label=f'Mean: ${np.mean(pnl_values):.2f}')
        axes[1, 0].set_title('Trade P&L Distribution')
        axes[1, 0].set_xlabel('P&L ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

    # Performance metrics
    metrics_text = ".2%"
    axes[1, 1].text(0.1, 0.9, metrics_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig

# Example strategy functions
def ml_prediction_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Simple strategy based on ML predictions
    Assumes 'prediction' column exists (1=bullish, 0=bearish)
    """
    df = data.copy()

    # Generate signals from predictions
    df['signal'] = 0
    df.loc[df['prediction'] == 1, 'signal'] = 1  # Buy signal
    df.loc[df['prediction'] == 0, 'signal'] = -1  # Sell signal

    return df

def moving_average_crossover_strategy(data: pd.DataFrame,
                                    fast_period: int = 10,
                                    slow_period: int = 20) -> pd.DataFrame:
    """
    Moving average crossover strategy
    """
    df = data.copy()

    # Calculate moving averages
    df['MA_fast'] = df['Close'].rolling(window=fast_period).mean()
    df['MA_slow'] = df['Close'].rolling(window=slow_period).mean()

    # Generate signals
    df['signal'] = 0
    df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1
    df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1

    return df

def rsi_strategy(data: pd.DataFrame, rsi_period: int = 14,
                oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    """
    RSI-based strategy
    """
    df = data.copy()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Generate signals
    df['signal'] = 0
    df.loc[df['RSI'] < oversold, 'signal'] = 1  # Buy when oversold
    df.loc[df['RSI'] > overbought, 'signal'] = -1  # Sell when overbought

    return df

if __name__ == "__main__":
    # Example usage
    print("XAUUSD Backtesting Framework")
    print("Run this module to test strategies on your dataset")
    print("See example usage in the docstrings above")