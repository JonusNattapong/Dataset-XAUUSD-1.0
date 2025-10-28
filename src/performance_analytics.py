#!/usr/bin/env python3
"""
XAUUSD Performance Analytics Module

This module provides comprehensive performance analysis, visualization,
and reporting capabilities for trading strategies and portfolios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our backtesting framework
try:
    from backtesting_framework import BacktestResult, Trade
    from risk_management import RiskMetrics
except ImportError:
    # Fallback for standalone usage
    pass

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    monthly_returns: pd.Series

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    value_at_risk: float

    # Trade metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float
    largest_win: float
    largest_loss: float

    # Risk-adjusted metrics
    kelly_criterion: float
    ulcer_index: float
    sterling_ratio: float

class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies"""

    def __init__(self, initial_capital: float = 100000, risk_free_rate: float = 0.02):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def analyze_backtest_results(self, results: 'BacktestResult') -> PerformanceMetrics:
        """
        Analyze backtest results and compute comprehensive metrics

        Args:
            results: BacktestResult from backtesting framework

        Returns:
            PerformanceMetrics object
        """

        # Extract data
        trades = results.trades
        equity_curve = results.equity_curve

        # Basic return metrics
        total_return = results.total_return
        annualized_return = results.annualized_return

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)

        # Risk metrics
        returns = equity_curve.pct_change().dropna()
        volatility = results.volatility

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(self.trading_days_per_year)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(self.trading_days_per_year)
        else:
            sortino_ratio = float('inf')

        # Maximum drawdown
        max_drawdown = results.max_drawdown

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')

        # Value at Risk (95% confidence, 1-day)
        value_at_risk = np.percentile(returns, 5)

        # Trade metrics
        total_trades = results.total_trades

        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]

            win_rate = len(winning_trades) / total_trades
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

            if avg_loss != 0:
                profit_factor = abs(sum([t.pnl for t in winning_trades]) / sum([t.pnl for t in losing_trades]))
            else:
                profit_factor = float('inf')

            avg_holding_period = np.mean([t.holding_period for t in trades])
            largest_win = max([t.pnl for t in trades]) if trades else 0
            largest_loss = min([t.pnl for t in trades]) if trades else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
            largest_win = 0
            largest_loss = 0

        # Advanced risk metrics
        kelly_criterion = self._calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        ulcer_index = self._calculate_ulcer_index(equity_curve)
        sterling_ratio = self._calculate_sterling_ratio(annualized_return, max_drawdown)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_returns=monthly_returns,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            value_at_risk=value_at_risk,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_period=avg_holding_period,
            largest_win=largest_win,
            largest_loss=largest_loss,
            kelly_criterion=kelly_criterion,
            ulcer_index=ulcer_index,
            sterling_ratio=sterling_ratio
        )

    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns from equity curve"""
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            # Assume daily frequency if no datetime index
            monthly_equity = equity_curve.groupby(equity_curve.index // 21).last()
        else:
            monthly_equity = equity_curve.resample('M').last()

        monthly_returns = monthly_equity.pct_change().dropna()
        return monthly_returns

    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if avg_loss == 0 or win_rate == 0:
            return 0

        # Kelly formula: (bp - q) / b
        # where b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p

        kelly = (b * p - q) / b
        return max(0, kelly)  # Don't go negative

    def _calculate_ulcer_index(self, equity_curve: pd.Series) -> float:
        """Calculate Ulcer Index (measure of downside volatility)"""
        # Calculate drawdowns
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        # Square the drawdowns and average
        ulcer = np.sqrt(np.mean(drawdown**2))
        return ulcer

    def _calculate_sterling_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Sterling Ratio (return per unit of drawdown)"""
        if max_drawdown == 0:
            return float('inf')

        # Sterling ratio uses average drawdown, but we'll use max for simplicity
        return annualized_return / abs(max_drawdown)

    def generate_performance_report(self, metrics: PerformanceMetrics,
                                  strategy_name: str = "Strategy") -> str:
        """
        Generate a comprehensive performance report

        Args:
            metrics: PerformanceMetrics object
            strategy_name: Name of the strategy

        Returns:
            Formatted performance report
        """

        report = f"""
{'='*60}
{strategy_name.upper()} PERFORMANCE REPORT
{'='*60}

PERFORMANCE SUMMARY
{'-'*30}
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
Volatility: {metrics.volatility:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.3f}
Sortino Ratio: {metrics.sortino_ratio:.3f}
Maximum Drawdown: {metrics.max_drawdown:.2%}
Calmar Ratio: {metrics.calmar_ratio:.3f}
Value at Risk (95%): {metrics.value_at_risk:.2%}

TRADE STATISTICS
{'-'*30}
Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate:.1%}
Average Win: ${metrics.avg_win:.2f}
Average Loss: ${metrics.avg_loss:.2f}
Profit Factor: {metrics.profit_factor:.3f}
Average Holding Period: {metrics.avg_holding_period:.1f} days
Largest Win: ${metrics.largest_win:.2f}
Largest Loss: ${metrics.largest_loss:.2f}

RISK METRICS
{'-'*30}
Kelly Criterion: {metrics.kelly_criterion:.3f}
Ulcer Index: {metrics.ulcer_index:.4f}
Sterling Ratio: {metrics.sterling_ratio:.3f}

MONTHLY RETURNS
{'-'*30}
"""
        # Add monthly returns summary
        if len(metrics.monthly_returns) > 0:
            monthly_summary = metrics.monthly_returns.describe()
            report += f"Mean Monthly Return: {monthly_summary['mean']:.2%}\n"
            report += f"Best Month: {metrics.monthly_returns.max():.2%}\n"
            report += f"Worst Month: {metrics.monthly_returns.min():.2%}\n"
            report += f"Monthly Win Rate: {(metrics.monthly_returns > 0).mean():.1%}\n"

        # Performance rating
        rating = self._calculate_performance_rating(metrics)
        report += f"""
PERFORMANCE RATING: {rating}
{'='*60}
"""

        return report

    def _calculate_performance_rating(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance rating"""

        score = 0

        # Return score (40% weight)
        if metrics.annualized_return > 0.30: score += 40
        elif metrics.annualized_return > 0.20: score += 30
        elif metrics.annualized_return > 0.10: score += 20
        elif metrics.annualized_return > 0.05: score += 10

        # Risk score (30% weight)
        if abs(metrics.max_drawdown) < 0.10: score += 30
        elif abs(metrics.max_drawdown) < 0.15: score += 20
        elif abs(metrics.max_drawdown) < 0.20: score += 10

        # Sharpe score (20% weight)
        if metrics.sharpe_ratio > 2.0: score += 20
        elif metrics.sharpe_ratio > 1.5: score += 15
        elif metrics.sharpe_ratio > 1.0: score += 10
        elif metrics.sharpe_ratio > 0.5: score += 5

        # Win rate score (10% weight)
        if metrics.win_rate > 0.60: score += 10
        elif metrics.win_rate > 0.55: score += 7
        elif metrics.win_rate > 0.50: score += 5

        # Determine rating
        if score >= 80: return "EXCELLENT"
        elif score >= 65: return "VERY GOOD"
        elif score >= 50: return "GOOD"
        elif score >= 35: return "FAIR"
        else: return "POOR"

    def create_performance_dashboard(self, metrics: PerformanceMetrics,
                                   equity_curve: pd.Series,
                                   trades: List['Trade'],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive performance dashboard

        Args:
            metrics: PerformanceMetrics object
            equity_curve: Portfolio equity curve
            trades: List of trades
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Trading Strategy Performance Dashboard', fontsize=16, fontweight='bold')

        # 1. Equity Curve
        axes[0, 0].plot(equity_curve.index, equity_curve.values, linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Equity Curve', fontweight='bold')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(equity_curve.index, equity_curve.values, self.initial_capital,
                               where=(equity_curve.values >= self.initial_capital),
                               color='green', alpha=0.3, label='Profit')
        axes[0, 0].fill_between(equity_curve.index, equity_curve.values, self.initial_capital,
                               where=(equity_curve.values < self.initial_capital),
                               color='red', alpha=0.3, label='Loss')
        axes[0, 0].legend()

        # 2. Drawdown Chart
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak * 100
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.7)
        axes[0, 1].set_title('Portfolio Drawdown', fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].set_ylim(drawdown.min()*1.1, 0)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Monthly Returns Heatmap
        if len(metrics.monthly_returns) > 0:
            monthly_pivot = self._create_monthly_returns_heatmap(metrics.monthly_returns)
            if monthly_pivot is not None:
                sns.heatmap(monthly_pivot, ax=axes[0, 2], cmap='RdYlGn', center=0,
                           annot=True, fmt='.1%', cbar_kws={'label': 'Return'})
                axes[0, 2].set_title('Monthly Returns Heatmap', fontweight='bold')
            else:
                axes[0, 2].text(0.5, 0.5, 'Insufficient data for\nmonthly heatmap',
                               ha='center', va='center', transform=axes[0, 2].transAxes)
                axes[0, 2].set_title('Monthly Returns', fontweight='bold')
        else:
            axes[0, 2].text(0.5, 0.5, 'No monthly data available',
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Monthly Returns', fontweight='bold')

        # 4. Trade P&L Distribution
        if trades:
            pnl_values = [t.pnl for t in trades]
            axes[1, 0].hist(pnl_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 0].axvline(np.mean(pnl_values), color='red', linestyle='--',
                             label=f'Mean: ${np.mean(pnl_values):.2f}')
            axes[1, 0].set_title('Trade P&L Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Performance Metrics Summary
        metrics_text = ".2%"
        axes[1, 1].text(0.1, 0.95, metrics_text, fontsize=10, verticalalignment='top',
                       fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Key Performance Metrics', fontweight='bold')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        # 6. Rolling Sharpe Ratio
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 30:
            rolling_sharpe = self._calculate_rolling_sharpe(returns, window=30)
            axes[1, 2].plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', linewidth=2)
            axes[1, 2].set_title('30-Day Rolling Sharpe Ratio', fontweight='bold')
            axes[1, 2].set_ylabel('Sharpe Ratio')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            axes[1, 2].legend()

        # 7. Trade Timing Analysis
        if trades:
            holding_periods = [t.holding_period for t in trades]
            axes[2, 0].hist(holding_periods, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[2, 0].axvline(np.mean(holding_periods), color='red', linestyle='--',
                             label=f'Mean: {np.mean(holding_periods):.1f} days')
            axes[2, 0].set_title('Trade Holding Periods', fontweight='bold')
            axes[2, 0].set_xlabel('Holding Period (days)')
            axes[2, 0].set_ylabel('Frequency')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # 8. Cumulative Returns Comparison
        buy_hold_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        strategy_return = metrics.total_return

        axes[2, 1].bar(['Buy & Hold', 'Strategy'], [buy_hold_return, strategy_return],
                      color=['gray', 'blue'], alpha=0.7)
        axes[2, 1].set_title('Return Comparison', fontweight='bold')
        axes[2, 1].set_ylabel('Total Return')
        axes[2, 1].grid(True, alpha=0.3)

        # Format y-axis as percentage
        axes[2, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # 9. Risk-Return Scatter (placeholder for multiple strategies)
        axes[2, 2].scatter([metrics.volatility], [metrics.annualized_return],
                          s=100, color='red', alpha=0.7)
        axes[2, 2].set_title('Risk-Return Profile', fontweight='bold')
        axes[2, 2].set_xlabel('Volatility (Annualized)')
        axes[2, 2].set_ylabel('Annualized Return')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")

        return fig

    def _create_monthly_returns_heatmap(self, monthly_returns: pd.Series) -> Optional[pd.DataFrame]:
        """Create monthly returns heatmap data"""
        if not isinstance(monthly_returns.index, pd.DatetimeIndex) or len(monthly_returns) < 12:
            return None

        # Create pivot table
        monthly_pivot = monthly_returns.groupby([
            monthly_returns.index.year,
            monthly_returns.index.month
        ]).first().unstack()

        return monthly_pivot

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        rolling_mean = excess_returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()

        rolling_sharpe = rolling_mean / rolling_std * np.sqrt(self.trading_days_per_year)
        return rolling_sharpe

    def compare_strategies(self, strategy_results: Dict[str, PerformanceMetrics]) -> pd.DataFrame:
        """
        Compare multiple strategies side by side

        Args:
            strategy_results: Dictionary of strategy names to PerformanceMetrics

        Returns:
            DataFrame with comparison metrics
        """

        comparison_data = {}

        for strategy_name, metrics in strategy_results.items():
            comparison_data[strategy_name] = {
                'Total Return': f"{metrics.total_return:.2%}",
                'Annualized Return': f"{metrics.annualized_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.1%}",
                'Profit Factor': f"{metrics.profit_factor:.3f}",
                'Total Trades': metrics.total_trades,
                'Calmar Ratio': f"{metrics.calmar_ratio:.3f}",
                'Sortino Ratio': f"{metrics.sortino_ratio:.3f}"
            }

        return pd.DataFrame(comparison_data).T

if __name__ == "__main__":
    # Example usage
    print("XAUUSD Performance Analytics Module")
    print("This module provides comprehensive performance analysis")
    print("See docstrings for usage examples")