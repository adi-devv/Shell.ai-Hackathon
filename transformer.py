import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import warnings
import os
import logging
import time

# Comprehensive warning suppression
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['XGBOOST_SILENT'] = '1'
os.environ['LIGHTGBM_SILENT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

pd.options.mode.chained_assignment = None
np.seterr(all='ignore')

logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('catboost').setLevel(logging.ERROR)

# Optimized Feature Engineering with vectorized operations
class OptimizedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        # Vectorized operations for speed
        fractions = np.array([X[f'{comp}_fraction'].values for comp in components])
        
        # Calculate weighted averages efficiently
        for prop in range(1, 11):
            props = np.array([X[f'{comp}_Property{prop}'].values for comp in components])
            
            # Vectorized weighted average
            X[f'WeightedAvg_Property{prop}'] = np.sum(fractions * props, axis=0)
            
            # Vectorized statistical measures
            X[f'StdDev_Property{prop}'] = np.std(props, axis=0)
            X[f'Range_Property{prop}'] = np.ptp(props, axis=0)  # peak-to-peak (max-min)
            X[f'Variance_Property{prop}'] = np.var(props, axis=0)
            
            # Distance-based features
            mean_props = np.mean(props, axis=0)
            X[f'Property{prop}_euclidean_dist'] = np.sqrt(np.sum((props - mean_props)**2, axis=0))
            
            # Weighted variance
            weighted_mean = np.sum(fractions * props, axis=0)
            X[f'WeightedVar_Property{prop}'] = np.sum(fractions * (props - weighted_mean)**2, axis=0)
        
        # Mixture entropy (vectorized)
        X['mixture_entropy'] = -np.sum(fractions * np.log(fractions + 1e-10), axis=0)
        X['mixture_complexity'] = 1 - np.sum(fractions**2, axis=0)
        
        # Dominant component features
        max_frac_idx = np.argmax(fractions, axis=0)
        X['dominant_fraction'] = np.max(fractions, axis=0)
        X['second_dominant_fraction'] = np.partition(fractions, -2, axis=0)[-2]
        X['dominance_ratio'] = X['dominant_fraction'] / (X['second_dominant_fraction'] + 1e-10)
        
        # Reduced component interactions (most important only)
        important_pairs = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4)]
        for i, j in important_pairs:
            comp_i = f'Component{i}_fraction'
            comp_j = f'Component{j}_fraction'
            X[f'Component{i}_Component{j}_interaction'] = X[comp_i] * X[comp_j]
            X[f'Component{i}_Component{j}_ratio'] = X[comp_i] / (X[comp_j] + 1e-10)
        
        # Reduced cross-property interactions (most important only)
        important_prop_pairs = [(1, 2), (1, 3), (2, 3), (1, 5), (2, 5), (3, 5)]
        for i, j in important_prop_pairs:
            X[f'WeightedAvg_Property{i}_Property{j}_ratio'] = X[f'WeightedAvg_Property{i}'] / (X[f'WeightedAvg_Property{j}'] + 1e-10)
            X[f'WeightedAvg_Property{i}_Property{j}_product'] = X[f'WeightedAvg_Property{i}'] * X[f'WeightedAvg_Property{j}']
        
        # Reduced polynomial features (most important only)
        important_props = [1, 2, 3, 5, 7]
        for prop in important_props:
            X[f'Property{prop}_squared'] = X[f'WeightedAvg_Property{prop}'] ** 2
            X[f'Property{prop}_sqrt'] = np.sqrt(np.abs(X[f'WeightedAvg_Property{prop}']))
            X[f'Property{prop}_log'] = np.log(np.abs(X[f'WeightedAvg_Property{prop}']) + 1e-10)
        
        return X

# Function to train a single model (for parallel processing)
def train_single_model(prop, X_train, y_train, X_val, y_val, target_transformers):
    """Train a single model for a specific property"""
    print(f"Training model for BlendProperty{prop}")
    
    # Special configuration for BlendProperty1 (the problematic one)
    if prop == 1:
        print("Using advanced ensemble for BlendProperty1...")
        # Ultra-robust ensemble for BlendProperty1
        base_models = [
            ('xgb', XGBRegressor(
                n_estimators=1500,
                max_depth=10,
                learning_rate=0.02,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=1.0,
                reg_lambda=3.0,
                gamma=0.1,
                min_child_weight=3,
                random_state=42,
                verbosity=0,
                n_jobs=1
            )),
            ('lgbm', LGBMRegressor(
                n_estimators=1500,
                max_depth=12,
                learning_rate=0.02,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=1.0,
                reg_lambda=3.0,
                min_child_samples=5,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )),
            ('catboost', CatBoostRegressor(
                iterations=1500,
                depth=10,
                learning_rate=0.02,
                l2_leaf_reg=7.0,
                random_state=42,
                silent=True,
                thread_count=1
            )),
            ('extra_trees', ExtraTreesRegressor(
                n_estimators=800,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42,
                n_jobs=1
            )),
            ('rf', RandomForestRegressor(
                n_estimators=800,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                random_state=42,
                n_jobs=1
            )),
            ('ridge', Ridge(
                alpha=10.0,
                random_state=42
            ))
        ]
        feature_engineer = AdvancedFeatureEngineer(for_property1=True)
    else:
        # Standard models for other properties
        base_models = [
            ('xgb', XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=1
            )),
            ('lgbm', LGBMRegressor(
                n_estimators=800,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=1
            )),
            ('catboost', CatBoostRegressor(
                iterations=800,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_state=42,
                silent=True,
                thread_count=1
            ))
        ]
        feature_engineer = OptimizedFeatureEngineer()
    
    # Preprocessing with robust scaling for BlendProperty1
    numeric_features = X_train.columns.tolist()
    if prop == 1:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # More robust to outliers
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Feature selection - more features for BlendProperty1
    if prop == 1:
        feature_selector = SelectKBest(score_func=f_regression, k=200)  # Even more features
    else:
        feature_selector = SelectKBest(score_func=f_regression, k=120)
    
    # Stacking configuration
    if prop == 1:
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=ElasticNet(alpha=0.01, l1_ratio=0.5),  # ElasticNet for regularization
            cv=7,  # More CV folds
            n_jobs=1
        )
    else:
        stacking_regressor = StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=0.1),
            cv=3,
            n_jobs=1
        )
    
    # Build pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineer', feature_engineer),
        ('preprocessor', preprocessor),
        ('feature_selector', feature_selector),
        ('regressor', stacking_regressor)
    ])
    
    # Train model
    start_time = time.time()
    pipeline.fit(X_train, y_train[f'BlendProperty{prop}'])
    training_time = time.time() - start_time
    
    # Validate
    val_pred = pipeline.predict(X_val)
    
    # Transform predictions back to original scale
    val_pred_original = target_transformers[f'BlendProperty{prop}'].inverse_transform(val_pred.reshape(-1, 1)).flatten()
    y_val_original = target_transformers[f'BlendProperty{prop}'].inverse_transform(y_val[f'BlendProperty{prop}'].values.reshape(-1, 1)).flatten()
    
    mape = mean_absolute_percentage_error(y_val_original, val_pred_original)
    print(f"BlendProperty{prop} - MAPE: {mape:.4f}, Training time: {training_time:.2f}s")
    
    return prop, pipeline, mape

def analyze_problem_property(X, y, prop_num):
    """Analyze why a specific property has high MAPE"""
    target_col = f'BlendProperty{prop_num}'
    
    print(f"\nAnalyzing {target_col}:")
    print(f"Target stats: min={y[target_col].min():.6f}, max={y[target_col].max():.6f}")
    print(f"Target mean: {y[target_col].mean():.6f}, std: {y[target_col].std():.6f}")
    print(f"Target distribution skewness: {y[target_col].skew():.4f}")
    
    # Check for outliers
    Q1 = y[target_col].quantile(0.25)
    Q3 = y[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = y[(y[target_col] < Q1 - 1.5*IQR) | (y[target_col] > Q3 + 1.5*IQR)]
    print(f"Outliers detected: {len(outliers)} ({len(outliers)/len(y)*100:.2f}%)")
    
    # Check for near-zero values (which can cause high MAPE)
    near_zero = y[y[target_col].abs() < 0.001]
    print(f"Near-zero values (< 0.001): {len(near_zero)} ({len(near_zero)/len(y)*100:.2f}%)")
    
    very_small = y[y[target_col].abs() < 0.01]
    print(f"Very small values (< 0.01): {len(very_small)} ({len(very_small)/len(y)*100:.2f}%)")
    
    small = y[y[target_col].abs() < 0.1]
    print(f"Small values (< 0.1): {len(small)} ({len(small)/len(y)*100:.2f}%)")
    
    # Distribution analysis
    print(f"25th percentile: {Q1:.6f}")
    print(f"50th percentile: {y[target_col].median():.6f}")
    print(f"75th percentile: {Q3:.6f}")
    print(f"99th percentile: {y[target_col].quantile(0.99):.6f}")
    
    return len(outliers), len(near_zero)

def create_robust_transformer_for_property1(y):
    """Create a robust transformer for BlendProperty1"""
    prop1_data = y['BlendProperty1']
    
    # Check percentage of very small values
    very_small_pct = len(prop1_data[prop1_data.abs() < 0.01]) / len(prop1_data)
    
    if very_small_pct > 0.1:  # More than 10% very small values
        print(f"BlendProperty1 has {very_small_pct*100:.1f}% very small values. Using robust transformation.")
        
        # Custom robust transformer
        class RobustLogTransformer(BaseEstimator, TransformerMixin):
            def __init__(self, offset=0.001):
                self.offset = offset
                
            def fit(self, X, y=None):
                return self
                
            def transform(self, X):
                # Add offset to handle near-zero values, then log transform
                X_shifted = X + self.offset
                return np.log(X_shifted)
                
            def inverse_transform(self, X):
                return np.exp(X) - self.offset
                
        return RobustLogTransformer()
    else:
        # Try quantile transformation for non-normal distributions
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='normal', random_state=42)

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering specifically for problematic properties"""
    def __init__(self, for_property1=False):
        self.for_property1 = for_property1
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        # Basic feature engineering
        fractions = np.array([X[f'{comp}_fraction'].values for comp in components])
        
        # Calculate weighted averages efficiently
        for prop in range(1, 11):
            props = np.array([X[f'{comp}_Property{prop}'].values for comp in components])
            
            # Basic features
            X[f'WeightedAvg_Property{prop}'] = np.sum(fractions * props, axis=0)
            X[f'StdDev_Property{prop}'] = np.std(props, axis=0)
            X[f'Range_Property{prop}'] = np.ptp(props, axis=0)
            X[f'Variance_Property{prop}'] = np.var(props, axis=0)
            
            # Advanced features for Property1
            if self.for_property1:
                # More sophisticated statistical measures
                X[f'Skewness_Property{prop}'] = pd.Series(props.T.tolist()).apply(lambda x: pd.Series(x).skew())
                X[f'Kurtosis_Property{prop}'] = pd.Series(props.T.tolist()).apply(lambda x: pd.Series(x).kurtosis())
                X[f'CoefVar_Property{prop}'] = X[f'StdDev_Property{prop}'] / (X[f'WeightedAvg_Property{prop}'] + 1e-10)
                
                # Robust statistics
                X[f'IQR_Property{prop}'] = np.percentile(props, 75, axis=0) - np.percentile(props, 25, axis=0)
                X[f'MedianAbsDev_Property{prop}'] = np.median(np.abs(props - np.median(props, axis=0)), axis=0)
                
                # Non-linear transformations
                X[f'Property{prop}_harmonic_mean'] = len(components) / np.sum(1/(props + 1e-10), axis=0)
                X[f'Property{prop}_geometric_mean'] = np.exp(np.mean(np.log(props + 1e-10), axis=0))
        
        # Mixture entropy and complexity
        X['mixture_entropy'] = -np.sum(fractions * np.log(fractions + 1e-10), axis=0)
        X['mixture_complexity'] = 1 - np.sum(fractions**2, axis=0)
        X['mixture_balance'] = np.sum(fractions**2, axis=0)
        
        # Dominant component features
        max_frac_idx = np.argmax(fractions, axis=0)
        X['dominant_fraction'] = np.max(fractions, axis=0)
        X['second_dominant_fraction'] = np.partition(fractions, -2, axis=0)[-2]
        X['dominance_ratio'] = X['dominant_fraction'] / (X['second_dominant_fraction'] + 1e-10)
        
        if self.for_property1:
            # Advanced mixture features
            X['mixture_dispersion'] = np.sum(fractions * (1 - fractions), axis=0)
            X['mixture_concentration'] = np.sum(fractions**3, axis=0)
            
            # All pairwise component interactions
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    comp_i = f'Component{i+1}_fraction'
                    comp_j = f'Component{j+1}_fraction'
                    X[f'Component{i+1}_Component{j+1}_interaction'] = X[comp_i] * X[comp_j]
                    X[f'Component{i+1}_Component{j+1}_ratio'] = X[comp_i] / (X[comp_j] + 1e-10)
                    X[f'Component{i+1}_Component{j+1}_diff'] = X[comp_i] - X[comp_j]
            
            # Property correlations and interactions
            for i in range(1, 11):
                for j in range(i+1, 11):
                    X[f'WeightedAvg_Property{i}_Property{j}_ratio'] = X[f'WeightedAvg_Property{i}'] / (X[f'WeightedAvg_Property{j}'] + 1e-10)
                    X[f'WeightedAvg_Property{i}_Property{j}_product'] = X[f'WeightedAvg_Property{i}'] * X[f'WeightedAvg_Property{j}']
                    X[f'WeightedAvg_Property{i}_Property{j}_diff'] = X[f'WeightedAvg_Property{i}'] - X[f'WeightedAvg_Property{j}']
        else:
            # Standard interactions for other properties
            important_pairs = [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4)]
            for i, j in important_pairs:
                comp_i = f'Component{i}_fraction'
                comp_j = f'Component{j}_fraction'
                X[f'Component{i}_Component{j}_interaction'] = X[comp_i] * X[comp_j]
                X[f'Component{i}_Component{j}_ratio'] = X[comp_i] / (X[comp_j] + 1e-10)
            
            important_prop_pairs = [(1, 2), (1, 3), (2, 3), (1, 5), (2, 5), (3, 5)]
            for i, j in important_prop_pairs:
                X[f'WeightedAvg_Property{i}_Property{j}_ratio'] = X[f'WeightedAvg_Property{i}'] / (X[f'WeightedAvg_Property{j}'] + 1e-10)
                X[f'WeightedAvg_Property{i}_Property{j}_product'] = X[f'WeightedAvg_Property{i}'] * X[f'WeightedAvg_Property{j}']
        
        # Polynomial features
        important_props = [1, 2, 3, 5, 7] if not self.for_property1 else list(range(1, 11))
        for prop in important_props:
            X[f'Property{prop}_squared'] = X[f'WeightedAvg_Property{prop}'] ** 2
            X[f'Property{prop}_sqrt'] = np.sqrt(np.abs(X[f'WeightedAvg_Property{prop}']))
            X[f'Property{prop}_log'] = np.log(np.abs(X[f'WeightedAvg_Property{prop}']) + 1e-10)
            
            if self.for_property1:
                X[f'Property{prop}_cubed'] = X[f'WeightedAvg_Property{prop}'] ** 3
                X[f'Property{prop}_inv'] = 1 / (X[f'WeightedAvg_Property{prop}'] + 1e-10)
                X[f'Property{prop}_exp'] = np.exp(X[f'WeightedAvg_Property{prop}'] / 10)  # Scaled to avoid overflow
        
        return X

def main():
    """Main function to run the training process"""
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')

    # Separate features and targets
    X = train_data.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
    y = train_data[[f'BlendProperty{i}' for i in range(1, 11)]]

    # Analyze BlendProperty1 specifically
    outliers_count, near_zero_count = analyze_problem_property(X, y, 1)

    # Target transformation with special handling for BlendProperty1
    print("Applying target transformation...")
    target_transformers = {}
    y_transformed = y.copy()
    
    for prop in range(1, 11):
        if prop == 1:
            # Special handling for BlendProperty1
            transformer = create_robust_transformer_for_property1(y)
        else:
            transformer = PowerTransformer(method='yeo-johnson')
        
        try:
            y_transformed[f'BlendProperty{prop}'] = transformer.fit_transform(y[[f'BlendProperty{prop}']])
            target_transformers[f'BlendProperty{prop}'] = transformer
        except Exception as e:
            print(f"Warning: Transform failed for BlendProperty{prop}, using identity transform: {e}")
            # Fallback to identity transform
            from sklearn.preprocessing import FunctionTransformer
            identity_transformer = FunctionTransformer()
            y_transformed[f'BlendProperty{prop}'] = identity_transformer.fit_transform(y[[f'BlendProperty{prop}']])
            target_transformers[f'BlendProperty{prop}'] = identity_transformer

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

    # Method 1: Use ThreadPoolExecutor instead of ProcessPoolExecutor
    print("Starting parallel training with ThreadPoolExecutor...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 10)) as executor:
        # Submit all training jobs
        future_to_prop = {
            executor.submit(train_single_model, prop, X_train, y_train, X_val, y_val, target_transformers): prop
            for prop in range(1, 11)
        }
        
        # Collect results
        models = {}
        mapes = {}
        
        for future in future_to_prop:
            prop, pipeline, mape = future.result()
            models[f'BlendProperty{prop}'] = pipeline
            mapes[f'BlendProperty{prop}'] = mape

    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time:.2f}s")
    print(f"Average MAPE: {np.mean(list(mapes.values())):.4f}")

    # Make predictions on test data
    print("Making predictions...")
    predictions = pd.DataFrame()
    predictions['ID'] = test_data['ID']

    for prop in range(1, 11):
        # Get transformed predictions
        pred_transformed = models[f'BlendProperty{prop}'].predict(test_data.drop(columns=['ID']))
        
        # Transform back to original scale
        pred_original = target_transformers[f'BlendProperty{prop}'].inverse_transform(pred_transformed.reshape(-1, 1)).flatten()
        
        predictions[f'BlendProperty{prop}'] = pred_original

    # Save submission
    predictions.to_csv('submission_fast.csv', index=False)
    print("Fast submission file saved as submission_fast.csv")

    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Average training time per model: {total_training_time/10:.2f}s")
    print(f"CPU cores used: {min(mp.cpu_count(), 10)}")
    for prop in range(1, 11):
        print(f"BlendProperty{prop} MAPE: {mapes[f'BlendProperty{prop}']:.4f}")

# Alternative implementation using joblib (Method 2)
def train_with_joblib():
    """Alternative implementation using joblib for parallel processing"""
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')

    # Separate features and targets
    X = train_data.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
    y = train_data[[f'BlendProperty{i}' for i in range(1, 11)]]

    # Target transformation
    print("Applying target transformation...")
    target_transformers = {}
    y_transformed = y.copy()
    for prop in range(1, 11):
        transformer = PowerTransformer(method='yeo-johnson')
        y_transformed[f'BlendProperty{prop}'] = transformer.fit_transform(y[[f'BlendProperty{prop}']])
        target_transformers[f'BlendProperty{prop}'] = transformer

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

    # Parallel training using joblib
    print("Starting parallel training with joblib...")
    start_time = time.time()

    results = Parallel(n_jobs=min(mp.cpu_count(), 10), backend='threading')(
        delayed(train_single_model)(prop, X_train, y_train, X_val, y_val, target_transformers)
        for prop in range(1, 11)
    )

    # Collect results
    models = {}
    mapes = {}
    for prop, pipeline, mape in results:
        models[f'BlendProperty{prop}'] = pipeline
        mapes[f'BlendProperty{prop}'] = mape

    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time:.2f}s")
    print(f"Average MAPE: {np.mean(list(mapes.values())):.4f}")

    # Make predictions on test data
    print("Making predictions...")
    predictions = pd.DataFrame()
    predictions['ID'] = test_data['ID']

    for prop in range(1, 11):
        # Get transformed predictions
        pred_transformed = models[f'BlendProperty{prop}'].predict(test_data.drop(columns=['ID']))
        
        # Transform back to original scale
        pred_original = target_transformers[f'BlendProperty{prop}'].inverse_transform(pred_transformed.reshape(-1, 1)).flatten()
        
        predictions[f'BlendProperty{prop}'] = pred_original

    # Save submission
    predictions.to_csv('submission_fast_joblib.csv', index=False)
    print("Fast submission file saved as submission_fast_joblib.csv")

    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Average training time per model: {total_training_time/10:.2f}s")
    print(f"CPU cores used: {min(mp.cpu_count(), 10)}")
    for prop in range(1, 11):
        print(f"BlendProperty{prop} MAPE: {mapes[f'BlendProperty{prop}']:.4f}")

if __name__ == '__main__':
    # This is the critical fix for Windows multiprocessing
    mp.freeze_support()
    
    # Choose your preferred method:
    # Method 1: ThreadPoolExecutor (recommended for ML tasks with scikit-learn)
    main()
    
    # Method 2: Joblib (uncomment to use instead)
    # train_with_joblib()