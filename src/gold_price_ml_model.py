import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class GoldPricePredictor:
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
        self.models = {}
        self.scaler = RobustScaler()  # Better for financial data with outliers
        self.feature_selector = None

    def load_data(self, filepath='XAUUSD_enhanced_ml_dataset_clean.csv'):
        """Load and prepare the enhanced dataset"""
        print("Loading enhanced dataset...")
        df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')

        # Remove rows with NaN targets
        df = df.dropna(subset=['Target_1d'])

        # Separate features and targets
        feature_cols = [col for col in df.columns if not col.startswith('Target') and col != 'Price_Change_1d_Pct' and col != 'Price_Change_5d_Pct']
        target_cols = ['Target_1d', 'Price_Change_1d_Pct']

        X = df[feature_cols]
        y = df[target_cols[1]]  # Predict actual price change percentage

        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Target: {target_cols[1]}")

        return X, y, feature_cols

    def feature_selection(self, X, y, k=50):
        """Select most important features"""
        print(f"Selecting top {k} features...")

        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()

        self.feature_selector = selector

        print(f"Selected features: {selected_features[:10]}...")  # Show first 10

        return X[selected_features], selected_features

    def create_time_series_split(self, X, y):
        """Create time series cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=int(len(X) * self.test_size))

        splits = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    def train_models(self, X, y):
        """Train multiple models"""
        print("Training models...")

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Time series cross-validation
        splits = self.create_time_series_split(X_scaled, y)

        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        }

        results = {}

        for name, model in models.items():
            print(f"Training {name}...")
            cv_scores = []

            for X_train, X_test, y_train, y_test in splits:
                # Train model
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                cv_scores.append({
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })

            # Average CV scores
            avg_scores = {
                'MAE': np.mean([s['MAE'] for s in cv_scores]),
                'RMSE': np.mean([s['RMSE'] for s in cv_scores]),
                'R2': np.mean([s['R2'] for s in cv_scores])
            }

            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'avg_scores': avg_scores
            }

            print(f"   {name} - MAE: {avg_scores['MAE']:.4f}, RMSE: {avg_scores['RMSE']:.4f}, R²: {avg_scores['R2']:.4f}")
        self.models = results
        return results

    def evaluate_models(self, results):
        """Evaluate and compare model performance"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            model_name: {
                'MAE': results[model_name]['avg_scores']['MAE'],
                'RMSE': results[model_name]['avg_scores']['RMSE'],
                'R2': results[model_name]['avg_scores']['R2']
            }
            for model_name in results.keys()
        }).T

        print(comparison.round(4))

        # Find best model
        best_model = comparison['R2'].idxmax()
        print(f"\n🏆 Best performing model: {best_model}")
        print(f"   R² Score: {comparison.loc[best_model, 'R2']:.4f}")
        return comparison

    def plot_feature_importance(self, model_name, feature_names, top_n=20):
        """Plot feature importance for tree-based models"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return

        model = self.models[model_name]['model']

        if hasattr(model, 'feature_importances_'):
            # Get feature importance
            importance = model.feature_importances_
            indices = np.argsort(importance)[-top_n:]

            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"Model {model_name} doesn't have feature_importances_ attribute")

    def predict_future(self, model_name, X_recent, days_ahead=5):
        """Make future predictions"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return

        model = self.models[model_name]['model']

        # Scale recent data
        X_scaled = self.scaler.transform(X_recent)

        # Make prediction
        prediction = model.predict(X_scaled[-1:])[0]

        print(f"\n🔮 {model_name} Prediction for next {days_ahead} days:")
        print(f"   Predicted change: {prediction:.4f}")

        # Calculate confidence interval (simplified)
        # In practice, you'd use prediction intervals or ensemble methods
        recent_predictions = model.predict(X_scaled[-30:])  # Last 30 days
        std_dev = np.std(recent_predictions)

        print(f"   Confidence interval: [{prediction - 1.96*std_dev:.4f}, {prediction + 1.96*std_dev:.4f}]")
        print(f"   Prediction std dev: {std_dev:.4f}")
        return prediction

def main():
    # Initialize predictor
    predictor = GoldPricePredictor(n_splits=3, test_size=0.15)  # Smaller test size for time series

    # Load data
    X, y, feature_names = predictor.load_data()

    # Feature selection
    X_selected, selected_features = predictor.feature_selection(X, y, k=20)

    # Train models
    results = predictor.train_models(X_selected, y)

    # Evaluate models
    comparison = predictor.evaluate_models(results)

    # Plot feature importance for best model
    best_model = comparison['R2'].idxmax()
    predictor.plot_feature_importance(best_model, selected_features)

    # Make future prediction
    predictor.predict_future(best_model, X_selected)

    print("\n✅ Model training and evaluation complete!")
    print("📊 Check the generated plots and comparison results")

if __name__ == "__main__":
    main()