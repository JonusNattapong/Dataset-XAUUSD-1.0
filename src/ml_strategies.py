#!/usr/bin/env python3
"""
XAUUSD ML-Based Trading Strategies

This module implements various machine learning-driven trading strategies
using the XAUUSD dataset predictions and technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MLStrategyConfig:
    """Configuration for ML-based strategies"""
    model_type: str = 'random_forest'
    prediction_threshold: float = 0.6
    confidence_filter: bool = True
    min_confidence: float = 0.7
    use_technical_filters: bool = True
    retrain_frequency: int = 100  # Retrain every N bars
    feature_selection: bool = True

class MLTradingStrategy:
    """Base class for ML-based trading strategies"""

    def __init__(self, config: MLStrategyConfig = None):
        self.config = config or MLStrategyConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.last_train_idx = 0
        self.performance_history = []

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model training/prediction

        Args:
            data: Raw market data

        Returns:
            DataFrame with engineered features
        """

        df = data.copy()

        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Volatility features
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + (std_20 * 2)
        df['BB_lower'] = sma_20 - (std_20 * 2)
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # Momentum indicators
        df['ROC_10'] = df['Close'].pct_change(periods=10)
        df['MOM_10'] = df['Close'] - df['Close'].shift(10)

        # Volume features (if available)
        if 'Volume' in df.columns:
            df['volume_ma_10'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_10']

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        # Target variable (future returns)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Drop NaN values
        df = df.dropna()

        return df

    def select_features(self, data: pd.DataFrame) -> List[str]:
        """
        Select most important features using feature importance

        Args:
            data: Data with features and target

        Returns:
            List of selected feature names
        """

        if not self.config.feature_selection:
            # Use all numeric columns except target
            features = [col for col in data.columns
                       if col != 'target' and col != 'Date' and data[col].dtype in ['float64', 'int64']]
            return features

        # Simple feature selection based on correlation and variance
        numeric_cols = [col for col in data.columns
                       if col not in ['target', 'Date'] and data[col].dtype in ['float64', 'int64']]

        # Remove low variance features
        variances = data[numeric_cols].var()
        high_var_features = variances[variances > variances.quantile(0.25)].index.tolist()

        # Remove highly correlated features
        corr_matrix = data[high_var_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        selected_features = [f for f in high_var_features if f not in to_drop]

        # Limit to top 30 features
        return selected_features[:30]

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train the ML model

        Args:
            data: Training data with features and target
        """

        # Prepare features
        feature_data = self.prepare_features(data)
        self.feature_columns = self.select_features(feature_data)

        X = feature_data[self.feature_columns]
        y = feature_data['target']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Initialize model
        if self.config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif self.config.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.config.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        print(f"  • Training accuracy: {train_score:.3f}")
        print(f"  • Test accuracy: {test_score:.3f}")

        self.last_train_idx = len(feature_data)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for new data

        Args:
            data: New data to predict on

        Returns:
            DataFrame with predictions and confidence scores
        """

        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Prepare features
        feature_data = self.prepare_features(data)

        # Ensure we have all required features
        missing_features = [f for f in self.feature_columns if f not in feature_data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        X = feature_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Add to dataframe
        feature_data['prediction'] = predictions
        feature_data['confidence'] = np.max(probabilities, axis=1)

        return feature_data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on ML predictions

        Args:
            data: Data with predictions

        Returns:
            DataFrame with trading signals
        """

        df = data.copy()

        # Initialize signals
        df['signal'] = 0
        df['signal_strength'] = 0

        # Basic signal generation
        bullish_mask = (df['prediction'] == 1) & (df['confidence'] >= self.config.min_confidence)
        bearish_mask = (df['prediction'] == 0) & (df['confidence'] >= self.config.min_confidence)

        df.loc[bullish_mask, 'signal'] = 1
        df.loc[bearish_mask, 'signal'] = -1

        df.loc[bullish_mask, 'signal_strength'] = df.loc[bullish_mask, 'confidence']
        df.loc[bearish_mask, 'signal_strength'] = df.loc[bearish_mask, 'confidence']

        # Apply technical filters if enabled
        if self.config.use_technical_filters:
            df = self.apply_technical_filters(df)

        return df

    def apply_technical_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply technical analysis filters to ML signals

        Args:
            data: Data with signals

        Returns:
            Filtered data
        """

        df = data.copy()

        # RSI filter - avoid buying when overbought, selling when oversold
        df['rsi_filter'] = 1
        df.loc[(df['signal'] == 1) & (df['RSI'] > 70), 'rsi_filter'] = 0  # Don't buy when overbought
        df.loc[(df['signal'] == -1) & (df['RSI'] < 30), 'rsi_filter'] = 0  # Don't sell when oversold

        # Trend filter - align with moving averages
        df['trend_filter'] = 1
        df.loc[(df['signal'] == 1) & (df['Close'] < df['SMA_20']), 'trend_filter'] = 0  # Don't buy below MA
        df.loc[(df['signal'] == -1) & (df['Close'] > df['SMA_20']), 'trend_filter'] = 0  # Don't sell above MA

        # Volatility filter - reduce signals in high volatility
        df['vol_filter'] = 1
        high_vol_mask = df['volatility_20'] > df['volatility_20'].quantile(0.8)
        df.loc[high_vol_mask, 'vol_filter'] = 0.5  # Reduce signal strength in high vol

        # Apply filters
        df.loc[df['rsi_filter'] == 0, 'signal'] = 0
        df.loc[df['trend_filter'] == 0, 'signal'] = 0
        df['signal_strength'] = df['signal_strength'] * df['rsi_filter'] * df['trend_filter'] * df['vol_filter']

        return df

    def update_model(self, new_data: pd.DataFrame) -> None:
        """
        Update model with new data (online learning)

        Args:
            new_data: New data to update model with
        """

        if len(new_data) - self.last_train_idx >= self.config.retrain_frequency:
            print(f"Retraining model at index {len(new_data)}")
            self.train_model(new_data)

class EnsembleMLStrategy(MLTradingStrategy):
    """Ensemble strategy combining multiple ML models"""

    def __init__(self, config: MLStrategyConfig = None):
        super().__init__(config)
        self.models = {}
        self.model_weights = {}

    def train_ensemble(self, data: pd.DataFrame) -> None:
        """
        Train ensemble of models

        Args:
            data: Training data
        """

        model_configs = [
            ('rf', 'random_forest'),
            ('gb', 'gradient_boosting'),
            ('lr', 'logistic_regression'),
            ('svm', 'svm')
        ]

        feature_data = self.prepare_features(data)
        self.feature_columns = self.select_features(feature_data)

        X = feature_data[self.feature_columns]
        y = feature_data['target']
        X_scaled = self.scaler.fit_transform(X)

        for name, model_type in model_configs:
            config = MLStrategyConfig(model_type=model_type)
            strategy = MLTradingStrategy(config)
            strategy.scaler = self.scaler
            strategy.feature_columns = self.feature_columns
            strategy.train_model(data)
            self.models[name] = strategy.model

        # Equal weights initially
        self.model_weights = {name: 1/len(model_configs) for name in self.models.keys()}

    def predict_ensemble(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble predictions

        Args:
            data: Data to predict on

        Returns:
            DataFrame with ensemble predictions
        """

        feature_data = self.prepare_features(data)
        X = feature_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)

        # Get predictions from all models
        predictions = {}
        probabilities = {}

        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)
            predictions[name] = pred
            probabilities[name] = prob

        # Ensemble prediction (weighted majority vote)
        ensemble_pred = np.zeros(len(X))
        ensemble_prob = np.zeros((len(X), 2))

        for i in range(len(X)):
            bullish_votes = 0
            bearish_votes = 0
            bullish_prob = 0
            bearish_prob = 0

            for name, weight in self.model_weights.items():
                if predictions[name][i] == 1:
                    bullish_votes += weight
                    bullish_prob += probabilities[name][i][1] * weight
                else:
                    bearish_votes += weight
                    bearish_prob += probabilities[name][i][0] * weight

            if bullish_votes > bearish_votes:
                ensemble_pred[i] = 1
                ensemble_prob[i] = [bearish_prob, bullish_prob]
            else:
                ensemble_pred[i] = 0
                ensemble_prob[i] = [bullish_prob, bearish_prob]

        feature_data['prediction'] = ensemble_pred
        feature_data['confidence'] = np.max(ensemble_prob, axis=1)

        return feature_data

class MLStrategyOptimizer:
    """Optimize ML strategy parameters"""

    def __init__(self, strategy_class=MLTradingStrategy):
        self.strategy_class = strategy_class

    def optimize_parameters(self, data: pd.DataFrame,
                          param_grid: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search

        Args:
            data: Historical data
            param_grid: Parameter grid to search

        Returns:
            Best parameters found
        """

        best_score = 0
        best_params = {}

        # Simple grid search (can be enhanced with cross-validation)
        from itertools import product

        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())

        for params in param_combinations:
            param_dict = dict(zip(param_names, params))

            # Create strategy with parameters
            config = MLStrategyConfig(**param_dict)
            strategy = self.strategy_class(config)

            try:
                # Train and evaluate
                strategy.train_model(data)
                predictions = strategy.predict(data)
                signals = strategy.generate_signals(predictions)

                # Simple scoring based on signal quality
                signal_changes = signals['signal'].diff().abs().sum()
                if signal_changes > 0:
                    score = len(signals[signals['signal'] != 0]) / signal_changes
                else:
                    score = 0

                if score > best_score:
                    best_score = score
                    best_params = param_dict

            except Exception as e:
                print(f"Error with params {param_dict}: {e}")
                continue

        print(f"Best parameters: {best_params}, Score: {best_score:.3f}")
        return best_params

if __name__ == "__main__":
    # Example usage
    print("XAUUSD ML-Based Trading Strategies")
    print("This module provides ML-driven trading strategies")
    print("See docstrings for usage examples")