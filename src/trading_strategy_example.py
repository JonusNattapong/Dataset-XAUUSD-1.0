#!/usr/bin/env python3
"""
XAUUSD Trading Strategy Example

This script demonstrates a complete workflow for developing and testing
a machine learning-based trading strategy using the XAUUSD dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from xauusd_dataset import load_xauusd_dataset
from backtesting_framework import Backtester, ml_prediction_strategy, moving_average_crossover_strategy
from risk_management import RiskManager
from ml_strategies import MLTradingStrategy, MLStrategyConfig
from performance_analytics import PerformanceAnalyzer

def main():
    """Main function demonstrating the complete trading workflow"""

    print("="*60)
    print("XAUUSD ML Trading Strategy Example")
    print("="*60)

    # 1. Load and prepare data
    print("\n1. Loading XAUUSD dataset...")
    try:
        data = load_xauusd_dataset()
        print(f"✓ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        print(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Split data
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size].copy()
    test_data = data[train_size:].copy()

    print(f"  Training data: {len(train_data)} samples")
    print(f"  Testing data: {len(test_data)} samples")

    # 2. Train ML strategy
    print("\n2. Training ML strategy...")
    config = MLStrategyConfig(
        model_type='random_forest',
        prediction_threshold=0.55,
        use_technical_filters=True,
        min_confidence=0.6
    )

    ml_strategy = MLTradingStrategy(config)
    ml_strategy.train_model(train_data)

    # Generate predictions and signals
    predictions_df = ml_strategy.predict(test_data)
    signals_df = ml_strategy.generate_signals(predictions_df)

    signals_count = len(signals_df[signals_df['signal'] != 0])
    buy_signals = len(signals_df[signals_df['signal'] == 1])
    sell_signals = len(signals_df[signals_df['signal'] == -1])

    print(f"✓ Generated {signals_count} trading signals")
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")

    # 3. Run backtest
    print("\n3. Running backtest...")
    initial_capital = 100000
    backtester = Backtester(signals_df, initial_capital)

    backtest_result = backtester.run_backtest(
        signals_df,
        position_size_pct=0.05,  # 5% of capital per trade
        stop_loss_pct=0.02,      # 2% stop loss
        take_profit_pct=0.04,    # 4% take profit
        max_holding_period=10    # Max 10 days
    )

    print(f"✓ Backtest completed: {backtest_result.total_trades} trades executed")

    # 4. Risk management example
    print("\n4. Risk management check...")
    risk_manager = RiskManager(
        initial_capital=initial_capital,
        max_portfolio_risk=0.02
    )

    # Example position sizing
    current_price = signals_df['Close'].iloc[-1]
    stop_loss_price = current_price * 0.98

    position_size = risk_manager.calculate_position_size(
        current_price, stop_loss_price, 0.02
    )

    approved, reason = risk_manager.check_risk_limits(
        position_size * current_price, position_size
    )

    print(".2f"    print(".2f"    print(f"  Risk per trade: ${(position_size * current_price * 0.02):.2f}")
    print(f"  Trade approved: {approved} ({reason})")

    # 5. Performance analysis
    print("\n5. Performance analysis...")
    analyzer = PerformanceAnalyzer(initial_capital=initial_capital)
    metrics = analyzer.analyze_backtest_results(backtest_result)

    print("✓ Key Performance Metrics:"    print(".2%"    print(".2%"    print(".3f"    print(".2%"    print(".1%"    print(".3f"
    # Determine rating
    if metrics.sharpe_ratio > 2.0 and metrics.total_return > 0.10:
        rating = "EXCELLENT"
    elif metrics.sharpe_ratio > 1.5 and metrics.total_return > 0.05:
        rating = "VERY GOOD"
    elif metrics.sharpe_ratio > 1.0:
        rating = "GOOD"
    elif metrics.total_return > 0:
        rating = "FAIR"
    else:
        rating = "NEEDS IMPROVEMENT"

    print(f"  Overall Rating: {rating}")

    # 6. Compare with benchmark
    print("\n6. Benchmark comparison...")

    # Buy and hold strategy
    buy_hold_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]

    # Moving average strategy
    ma_signals = moving_average_crossover_strategy(test_data)
    ma_backtester = Backtester(ma_signals, initial_capital)
    ma_result = ma_backtester.run_backtest(ma_signals)
    ma_metrics = analyzer.analyze_backtest_results(ma_result)

    print("✓ Strategy Comparison:"    print(".2%"    print(".2%"    print(".2%"    print(".3f"    print(".3f"    print(".3f"
    # 7. Generate simple performance plot
    print("\n7. Generating performance plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Equity curve
    ax1.plot(backtest_result.equity_curve.index, backtest_result.equity_curve.values,
             linewidth=2, label='ML Strategy')
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Drawdown
    peak = backtest_result.equity_curve.expanding().max()
    drawdown = (backtest_result.equity_curve - peak) / peak * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.7)
    ax2.set_title('Portfolio Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(drawdown.min()*1.1, 0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = Path("../results/ml_strategy_performance.png")
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance plot saved to {plot_path}")

    plt.close()

    # 8. Generate summary report
    print("\n8. Generating summary report...")

    report = f"""
{'='*60}
XAUUSD ML TRADING STRATEGY SUMMARY REPORT
{'='*60}

STRATEGY CONFIGURATION
{'-'*30}
Model Type: {config.model_type.upper()}
Prediction Threshold: {config.prediction_threshold}
Use Technical Filters: {config.use_technical_filters}
Minimum Confidence: {config.min_confidence}

DATASET INFO
{'-'*30}
Total Samples: {len(data)}
Training Samples: {len(train_data)}
Testing Samples: {len(test_data)}
Date Range: {data['Date'].min()} to {data['Date'].max()}

TRADING SIGNALS
{'-'*30}
Total Signals: {signals_count}
Buy Signals: {buy_signals}
Sell Signals: {sell_signals}
Signal Density: {signals_count/len(signals_df):.1%}

BACKTEST RESULTS
{'-'*30}
Initial Capital: ${initial_capital:,.0f}
Final Capital: ${backtest_result.equity_curve.iloc[-1]:,.2f}
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
Total Trades: {backtest_result.total_trades}
Win Rate: {metrics.win_rate:.1%}
Profit Factor: {metrics.profit_factor:.3f}

RISK METRICS
{'-'*30}
Volatility: {metrics.volatility:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.3f}
Sortino Ratio: {metrics.sortino_ratio:.3f}
Maximum Drawdown: {metrics.max_drawdown:.2%}
Calmar Ratio: {metrics.calmar_ratio:.3f}
Value at Risk (95%): {metrics.value_at_risk:.2%}

TRADE STATISTICS
{'-'*30}
Average Win: ${metrics.avg_win:.2f}
Average Loss: ${metrics.avg_loss:.2f}
Largest Win: ${metrics.largest_win:.2f}
Largest Loss: ${metrics.largest_loss:.2f}
Average Holding Period: {metrics.avg_holding_period:.1f} days

BENCHMARK COMPARISON
{'-'*30}
Buy & Hold Return: {buy_hold_return:.2%}
MA Crossover Return: {ma_metrics.total_return:.2%}
Strategy Outperformance: {metrics.total_return - buy_hold_return:.2%}

PERFORMANCE RATING: {rating}
{'='*60}
"""

    # Save report
    report_path = Path("../results/ml_strategy_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✓ Summary report saved to {report_path}")

    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nKey files generated:")
    print(f"- Performance plot: {plot_path}")
    print(f"- Summary report: {report_path}")
    print("\nNext steps:")
    print("1. Review the performance plot and report")
    print("2. Experiment with different strategy parameters")
    print("3. Try ensemble strategies for better performance")
    print("4. Run the full tutorial notebook for detailed analysis")

if __name__ == "__main__":
    main()