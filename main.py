import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Enhanced Feature Engineering
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.important_props = [1, 2, 3, 5, 7]  # Identified as important
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        # Feature storage for efficient concatenation
        new_features = {}
        
        # 1. Basic weighted features
        for prop in range(1, 11):
            weighted_sum = sum(X[f'{comp}_fraction'] * X[f'{comp}_Property{prop}'] for comp in components)
            new_features[f'WeightedAvg_Property{prop}'] = weighted_sum
            
            props = np.array([X[f'{comp}_Property{prop}'] for comp in components]).T
            new_features[f'StdDev_Property{prop}'] = np.std(props, axis=1)
            new_features[f'Range_Property{prop}'] = np.max(props, axis=1) - np.min(props, axis=1)
            new_features[f'Max_Property{prop}'] = np.max(props, axis=1)
            new_features[f'Min_Property{prop}'] = np.min(props, axis=1)
        
        # 2. Enhanced interaction features
        for i in range(1, 6):
            for j in range(i+1, 6):
                frac_i = f'Component{i}_fraction'
                frac_j = f'Component{j}_fraction'
                new_features[f'Frac_Interaction_{i}_{j}'] = X[frac_i] * X[frac_j]
                
                for prop in self.important_props:
                    prop_i = f'Component{i}_Property{prop}'
                    prop_j = f'Component{j}_Property{prop}'
                    new_features[f'Prop_Interaction_{prop}_{i}_{j}'] = X[prop_i] * X[prop_j]
                    new_features[f'Prop_Diff_{prop}_{i}_{j}'] = X[prop_i] - X[prop_j]
        
        # 3. Polynomial and non-linear features
        for prop in self.important_props:
            weighted = new_features[f'WeightedAvg_Property{prop}']
            new_features[f'Property{prop}_squared'] = weighted ** 2
            new_features[f'Property{prop}_sqrt'] = np.sqrt(np.abs(weighted))
            new_features[f'Property{prop}_log'] = np.log1p(np.abs(weighted))
            new_features[f'Property{prop}_reciprocal'] = 1 / (1 + np.abs(weighted))
        
        # 4. Composition features
        total_fractions = sum(X[f'{comp}_fraction'] for comp in components)
        for comp in components:
            new_features[f'{comp}_dominance'] = X[f'{comp}_fraction'] / (total_fractions + 1e-6)
        
        # Combine all features efficiently
        new_features_df = pd.DataFrame(new_features)
        return pd.concat([X, new_features_df], axis=1)

# Load data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Separate features and targets
X = train_data.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
y = train_data[[f'BlendProperty{i}' for i in range(1, 11)]]

# Enhanced data splitting with KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(X))

# Advanced preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('transformer', PowerTransformer(method='yeo-johnson'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.columns)
    ])

# Enhanced base models
def get_base_models():
    return [
        ('xgb', XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42,
            n_jobs=-1
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )),
        ('catboost', CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_seed=42,
            silent=True,
            thread_count=-1
        )),
        ('kernel_ridge', KernelRidge(
            alpha=0.5,
            kernel='polynomial',
            degree=2
        )),
        ('elasticnet', ElasticNet(
            alpha=0.001,
            l1_ratio=0.7,
            random_state=42,
            max_iter=2000
        ))
    ]

# Meta-models
def get_meta_models():
    return [
        ('ridge', Ridge(alpha=1.0)),
        ('lasso', Lasso(alpha=0.001)),
        ('svr', SVR(C=5, kernel='rbf'))
    ]

# Enhanced stacking approach
def create_enhanced_stacker():
    base_models = get_base_models()
    meta_models = get_meta_models()
    
    # Create multiple stacking layers
    level1 = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    level2 = StackingRegressor(
        estimators=meta_models,
        final_estimator=VotingRegressor([
            ('ridge', Ridge(alpha=0.5)),
            ('lgbm', LGBMRegressor(n_estimators=200))
        ]),
        cv=3,
        n_jobs=-1
    )
    
    return Pipeline([
        ('feature', AdvancedFeatureEngineer()),
        ('preprocess', preprocessor),
        ('feature_select', SelectFromModel(LGBMRegressor(), threshold='1.25*median')),
        ('pca', PCA(n_components=0.95)),
        ('regressor', level2)
    ])

# Train and predict for each property
models = {}
for prop in range(1, 11):
    print(f"\n=== Training BlendProperty{prop} ===")
    
    # Create enhanced model
    model = create_enhanced_stacker()
    
    # Cross-validation
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx][f'BlendProperty{prop}'], y.iloc[val_idx][f'BlendProperty{prop}']
        
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, val_pred)
        fold_scores.append(mape)
        print(f"Fold {fold+1} MAPE: {mape:.4f}")
    
    print(f"Average MAPE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    # Full training
    model.fit(X, y[f'BlendProperty{prop}'])
    models[f'BlendProperty{prop}'] = model

# Generate predictions
predictions = pd.DataFrame({'ID': test_data['ID']})
for prop in range(1, 11):
    predictions[f'BlendProperty{prop}'] = models[f'BlendProperty{prop}'].predict(test_data.drop(columns=['ID']))

# Save submission
predictions.to_csv('enhanced_submission.csv', index=False)
print("\nEnhanced submission saved!")