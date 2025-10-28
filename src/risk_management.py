#!/usr/bin/env python3
"""
XAUUSD Risk Management Module

This module provides comprehensive risk management tools for trading strategies,
including position sizing, stop-loss management, risk limits, and portfolio protection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    value_at_risk: float  # VaR at 95% confidence
    expected_shortfall: float  # CVaR at 95% confidence
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

@dataclass
class PositionLimits:
    """Position size limits"""
    max_position_size: float  # Maximum position size as % of portfolio
    max_sector_exposure: float  # Maximum exposure to single asset
    max_leverage: float  # Maximum leverage allowed
    max_concentration: float  # Maximum concentration in single position

class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, initial_capital: float = 100000,
                 max_portfolio_risk: float = 0.02,  # 2% max risk per trade
                 max_daily_loss: float = 0.05,     # 5% max daily loss
                 max_drawdown_limit: float = 0.10):  # 10% max drawdown

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.max_drawdown_limit = max_drawdown_limit

        # Risk tracking
        self.daily_pnl = 0
        self.peak_capital = initial_capital
        self.current_drawdown = 0

        # Position tracking
        self.open_positions = {}
        self.position_limits = PositionLimits(
            max_position_size=0.1,  # 10% of portfolio
            max_sector_exposure=0.2,  # 20% sector exposure
            max_leverage=5.0,  # 5x leverage max
            max_concentration=0.15  # 15% single position
        )

    def calculate_position_size(self, entry_price: float,
                              stop_loss_price: float,
                              risk_per_trade_pct: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management rules

        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            risk_per_trade_pct: Risk per trade as % of capital (overrides default)

        Returns:
            Position size in units
        """

        if risk_per_trade_pct is None:
            risk_per_trade_pct = self.max_portfolio_risk

        # Calculate risk amount
        risk_amount = self.current_capital * risk_per_trade_pct

        # Calculate stop loss distance
        if entry_price > stop_loss_price:  # Long position
            stop_distance = entry_price - stop_loss_price
        else:  # Short position
            stop_distance = stop_loss_price - entry_price

        if stop_distance == 0:
            return 0  # Avoid division by zero

        # Calculate position size
        position_value = risk_amount / (stop_distance / entry_price)

        # Apply position limits
        max_position_value = self.current_capital * self.position_limits.max_position_size
        position_value = min(position_value, max_position_value)

        # Convert to units
        position_size = position_value / entry_price

        return position_size

    def calculate_stop_loss(self, entry_price: float,
                          direction: str,
                          atr: Optional[float] = None,
                          risk_multiplier: float = 1.5) -> float:
        """
        Calculate stop loss price using ATR or percentage-based method

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            atr: Average True Range (optional)
            risk_multiplier: Multiplier for ATR-based stops

        Returns:
            Stop loss price
        """

        if atr is not None and atr > 0:
            # ATR-based stop loss
            if direction == 'long':
                stop_loss = entry_price - (atr * risk_multiplier)
            else:  # short
                stop_loss = entry_price + (atr * risk_multiplier)
        else:
            # Percentage-based stop loss (default 2%)
            stop_pct = 0.02
            if direction == 'long':
                stop_loss = entry_price * (1 - stop_pct)
            else:  # short
                stop_loss = entry_price * (1 + stop_pct)

        return stop_loss

    def calculate_take_profit(self, entry_price: float,
                            stop_loss_price: float,
                            risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price based on risk-reward ratio

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_reward_ratio: Desired risk-reward ratio

        Returns:
            Take profit price
        """

        risk_amount = abs(entry_price - stop_loss_price)
        reward_amount = risk_amount * risk_reward_ratio

        if entry_price > stop_loss_price:  # Long position
            take_profit = entry_price + reward_amount
        else:  # Short position
            take_profit = entry_price - reward_amount

        return take_profit

    def check_risk_limits(self, proposed_trade_value: float,
                         proposed_position_size: float) -> Tuple[bool, str]:
        """
        Check if a proposed trade violates risk limits

        Args:
            proposed_trade_value: Value of proposed trade
            proposed_position_size: Size of proposed position

        Returns:
            Tuple of (approved: bool, reason: str)
        """

        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * self.initial_capital:
            return False, "Daily loss limit exceeded"

        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_limit:
            return False, "Maximum drawdown limit exceeded"

        # Check position size limit
        if proposed_trade_value > self.current_capital * self.position_limits.max_position_size:
            return False, "Position size exceeds limit"

        # Check concentration limit
        total_exposure = sum([pos['value'] for pos in self.open_positions.values()])
        if total_exposure + proposed_trade_value > self.current_capital * self.position_limits.max_concentration:
            return False, "Portfolio concentration limit exceeded"

        # Check leverage limit
        total_leverage = (total_exposure + proposed_trade_value) / self.current_capital
        if total_leverage > self.position_limits.max_leverage:
            return False, "Leverage limit exceeded"

        return True, "Trade approved"

    def update_portfolio_risk(self, pnl: float) -> None:
        """
        Update portfolio risk metrics after a trade

        Args:
            pnl: Profit/Loss from the trade
        """

        self.current_capital += pnl
        self.daily_pnl += pnl

        # Update drawdown
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

    def calculate_portfolio_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for the portfolio

        Args:
            returns: Series of portfolio returns

        Returns:
            RiskMetrics object with all risk measures
        """

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Expected Shortfall (CVaR at 95%)
        es_95 = returns[returns <= var_95].mean()

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        if volatility > 0:
            sharpe = excess_returns.mean() / volatility * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = excess_returns.mean() / downside_vol * np.sqrt(252) if downside_vol > 0 else 0
        else:
            sortino = 0

        # Calmar Ratio
        annual_return = (1 + returns.mean())**252 - 1
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        return RiskMetrics(
            value_at_risk=var_95,
            expected_shortfall=es_95,
            max_drawdown=max_dd,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar
        )

    def get_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """
        Classify portfolio risk level based on metrics

        Args:
            metrics: RiskMetrics object

        Returns:
            RiskLevel enum value
        """

        # Risk assessment based on multiple factors
        risk_score = 0

        # Volatility assessment
        if metrics.volatility > 0.30:  # >30% volatility
            risk_score += 3
        elif metrics.volatility > 0.20:  # >20% volatility
            risk_score += 2
        elif metrics.volatility > 0.15:  # >15% volatility
            risk_score += 1

        # Drawdown assessment
        if abs(metrics.max_drawdown) > 0.20:  # >20% drawdown
            risk_score += 3
        elif abs(metrics.max_drawdown) > 0.15:  # >15% drawdown
            risk_score += 2
        elif abs(metrics.max_drawdown) > 0.10:  # >10% drawdown
            risk_score += 1

        # Sharpe ratio assessment (lower is riskier)
        if metrics.sharpe_ratio < 0.5:
            risk_score += 2
        elif metrics.sharpe_ratio < 1.0:
            risk_score += 1

        # Classify risk level
        if risk_score >= 5:
            return RiskLevel.EXTREME
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def generate_risk_report(self, metrics: RiskMetrics) -> str:
        """
        Generate a human-readable risk report

        Args:
            metrics: RiskMetrics object

        Returns:
            Formatted risk report string
        """

        risk_level = self.get_risk_level(metrics)

        report = ".2%"
        report += f"Risk Level: {risk_level.value.upper()}\n"
        report += f"Portfolio Value: ${self.current_capital:,.2f}\n"
        report += f"Current Drawdown: {self.current_drawdown:.2%}\n"
        report += f"Daily P&L: ${self.daily_pnl:,.2f}\n\n"

        # Risk warnings
        warnings = []
        if abs(metrics.max_drawdown) > 0.15:
            warnings.append("⚠️  High drawdown detected")
        if metrics.volatility > 0.25:
            warnings.append("⚠️  High volatility detected")
        if metrics.sharpe_ratio < 0.5:
            warnings.append("⚠️  Poor risk-adjusted returns")
        if self.current_drawdown > 0.10:
            warnings.append("⚠️  Current drawdown exceeds 10%")

        if warnings:
            report += "Risk Warnings:\n" + "\n".join(warnings) + "\n\n"

        # Recommendations
        if risk_level == RiskLevel.EXTREME:
            report += "Recommendations:\n"
            report += "• Consider reducing position sizes\n"
            report += "• Implement stricter stop-loss rules\n"
            report += "• Review strategy parameters\n"
            report += "• Consider portfolio rebalancing"
        elif risk_level == RiskLevel.HIGH:
            report += "Recommendations:\n"
            report += "• Monitor positions closely\n"
            report += "• Consider taking profits\n"
            report += "• Review risk management rules"

        return report

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for volatility measurement

    Args:
        data: DataFrame with OHLC data
        period: Period for ATR calculation

    Returns:
        Series with ATR values
    """

    high = data['High']
    low = data['Low']
    close = data['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Average True Range
    atr = tr.rolling(window=period).mean()

    return atr

if __name__ == "__main__":
    # Example usage
    print("XAUUSD Risk Management Module")
    print("This module provides comprehensive risk management tools")
    print("See docstrings for usage examples")