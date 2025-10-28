import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class AdvancedGoldPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.best_model = None

    def load_and_prepare_data(self):
        """Load cleaned dataset and prepare for advanced modeling"""
        print("Loading and preparing data for advanced modeling...")

        df = pd.read_csv('XAUUSD_enhanced_ml_dataset_clean.csv', parse_dates=['Date'], index_col='Date')

        # Focus on most relevant features for gold price prediction
        core_features = [
            # Price data
            'Open', 'High', 'Low', 'Close', 'Volume',

            # Key technical indicators
            'trend_sma_fast', 'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
            'momentum_rsi', 'momentum_stoch', 'momentum_stoch_signal',
            'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
            'trend_macd', 'trend_macd_signal', 'trend_macd_diff',

            # Economic indicators
            'DXY', 'DXY_Change', 'US_10Y_Yield', 'WTI_Oil', 'Silver',

            # Lagged features (recent history)
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5',

            # Rolling statistics
            'Rolling_Mean_5', 'Rolling_Mean_20', 'Rolling_Std_5', 'Rolling_Std_20',

            # Momentum
            'Momentum_1M', 'ROC_5', 'ROC_20',

            # Risk metrics
            'VaR_95', 'Sharpe_Ratio'
        ]

        # Filter to available features
        available_features = [f for f in core_features if f in df.columns]
        print(f"Using {len(available_features)} core features out of {len(core_features)} requested")

        X = df[available_features]
        y = df['Price_Change_1d_Pct']

        # Remove any remaining NaN
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, available_features

    def create_ensemble_model(self):
        """Create an ensemble model combining multiple algorithms"""
        print("Creating ensemble model...")

        # Define base models
        ridge = Ridge(alpha=0.1)
        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        lgb_model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)

        # Create ensemble
        ensemble = VotingRegressor([
            ('ridge', ridge),
            ('rf', rf),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ])

        return ensemble

    def advanced_cross_validation(self, X, y):
        """Perform advanced time series cross-validation"""
        print("Performing advanced time series cross-validation...")

        # Multiple train-test splits with expanding window
        tscv = TimeSeriesSplit(n_splits=5, test_size=30)  # 30-day test periods

        ensemble = self.create_ensemble_model()

        cv_scores = []
        feature_importance_list = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"   Fold {fold + 1}/5...")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Scale data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train ensemble
            ensemble.fit(X_train_scaled, y_train)

            # Predict
            y_pred = ensemble.predict(X_test_scaled)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # Directional accuracy (important for trading)
            directional_acc = np.mean((np.sign(y_pred) == np.sign(y_test)).astype(int))

            cv_scores.append({
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Directional_Accuracy': directional_acc
            })

            print(f"   Fold {fold + 1}: MAE={mae:.4f}, R²={r2:.4f}, Dir_Acc={directional_acc:.1%}")
        # Average scores
        avg_scores = {
            'MAE': np.mean([s['MAE'] for s in cv_scores]),
            'RMSE': np.mean([s['RMSE'] for s in cv_scores]),
            'R2': np.mean([s['R2'] for s in cv_scores]),
            'Directional_Accuracy': np.mean([s['Directional_Accuracy'] for s in cv_scores])
        }

        print("\n📊 Average CV Scores:")
        print(f"   MAE: {avg_scores['MAE']:.4f}")
        print(f"   RMSE: {avg_scores['RMSE']:.4f}")
        print(f"   R²: {avg_scores['R2']:.4f}")
        print(f"   Directional Accuracy: {avg_scores['Directional_Accuracy']:.4f}")
        return ensemble, avg_scores

    def train_final_model(self, X, y):
        """Train final model on all available data"""
        print("Training final model on all data...")

        ensemble = self.create_ensemble_model()

        # Scale all data
        X_scaled = self.scaler.fit_transform(X)

        # Train on all data
        ensemble.fit(X_scaled, y)

        self.best_model = ensemble
        self.scaler_fitted = True

        print("✅ Final model trained successfully!")
        return ensemble

    def predict_with_confidence(self, X_recent, confidence_level=0.95):
        """Make predictions with confidence intervals"""
        if self.best_model is None:
            print("No trained model available")
            return None

        # Scale recent data
        X_scaled = self.scaler.transform(X_recent)

        # Make prediction
        prediction = self.best_model.predict(X_scaled[-1:])[0]

        # Estimate uncertainty using recent predictions
        recent_predictions = self.best_model.predict(X_scaled[-60:])  # Last 60 days
        std_dev = np.std(recent_predictions)

        # Confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99% confidence
        ci_lower = prediction - z_score * std_dev
        ci_upper = prediction + z_score * std_dev

        return {
            'prediction': prediction,
            'confidence_interval': (ci_lower, ci_upper),
            'std_dev': std_dev
        }

    def analyze_feature_importance(self, X, feature_names):
        """Analyze feature importance across ensemble components"""
        print("Analyzing feature importance...")

        if not hasattr(self.best_model, 'estimators_'):
            print("Ensemble model not available for feature importance")
            return

        # Get feature importance from each model in ensemble
        importance_dict = {}

        for name, estimator in zip(self.best_model.named_estimators_.keys(), self.best_model.estimators_):
            if hasattr(estimator, 'feature_importances_'):
                importance_dict[name] = estimator.feature_importances_

        # Average importance across models
        if importance_dict:
            avg_importance = np.mean(list(importance_dict.values()), axis=0)

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': avg_importance
            }).sort_values('importance', ascending=False)

            # Plot top 15 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Average Importance')
            plt.title('Top 15 Feature Importance (Ensemble Average)')
            plt.tight_layout()
            plt.savefig('ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            print("📊 Top 10 most important features:")
            for i, row in importance_df.head(10).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

            return importance_df

    def create_trading_signals(self, predictions, threshold=0.005):
        """Convert predictions to trading signals"""
        signals = []

        for pred in predictions:
            if pred > threshold:
                signals.append('BUY')
            elif pred < -threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')

        return signals

def main():
    # Initialize advanced predictor
    predictor = AdvancedGoldPredictor()

    # Load and prepare data
    X, y, feature_names = predictor.load_and_prepare_data()

    # Perform advanced cross-validation
    ensemble_model, cv_scores = predictor.advanced_cross_validation(X, y)

    # Train final model
    final_model = predictor.train_final_model(X, y)

    # Analyze feature importance
    importance_df = predictor.analyze_feature_importance(X, feature_names)

    # Make prediction for next day
    recent_result = predictor.predict_with_confidence(X.tail(1))

    if recent_result:
        print("\n🔮 Next Day Prediction:")
        print(f"   Predicted change: {recent_result['prediction']:.4f}")
        print(f"   Confidence interval: [{recent_result['confidence_interval'][0]:.4f}, {recent_result['confidence_interval'][1]:.4f}]")
        print(f"   Prediction std dev: {recent_result['std_dev']:.4f}")
        # Generate trading signal
        signal = predictor.create_trading_signals([recent_result['prediction']])[0]
        print(f"📈 Trading Signal: {signal}")

    print("\n✅ Advanced modeling complete!")
    print("💡 Key Insights:")
    print(f"   - Directional Accuracy: {cv_scores['Directional_Accuracy']:.1%}")
    print("   - Focus on momentum and economic indicators")
    print("   - Consider ensemble methods for robustness")

if __name__ == "__main__":
    main()