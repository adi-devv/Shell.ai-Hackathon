import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
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

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        # Calculate weighted average of each property across components
        for prop in range(1, 11):
            weighted_props = []
            prop_cols = [f'{comp}_Property{prop}' for comp in components]
            frac_cols = [f'{comp}_fraction' for comp in components]
            
            for comp, frac in zip(components, frac_cols):
                weighted_props.append(X[frac] * X[f'{comp}_Property{prop}'])
            
            X[f'WeightedAvg_Property{prop}'] = sum(weighted_props)
            
            # Calculate standard deviation of properties across components
            props = np.array([X[f'{comp}_Property{prop}'] for comp in components]).T
            X[f'StdDev_Property{prop}'] = np.std(props, axis=1)
            
            # Calculate max-min difference of properties across components
            X[f'Range_Property{prop}'] = np.max(props, axis=1) - np.min(props, axis=1)
        
        # Interaction features between components
        for i in range(1, 6):
            for j in range(i+1, 6):
                comp_i = f'Component{i}_fraction'
                comp_j = f'Component{j}_fraction'
                X[f'Component{i}_Component{j}_interaction'] = X[comp_i] * X[comp_j]
                
                for prop in range(1, 6):
                    prop_i = f'Component{i}_Property{prop}'
                    prop_j = f'Component{j}_Property{prop}'
                    X[f'Component{i}_Property{prop}_Component{j}_Property{prop}_interaction'] = (
                        X[prop_i] * X[prop_j])
        
        # Polynomial features for important properties
        for prop in [1, 2, 3, 5, 7]:
            X[f'Property{prop}_squared'] = X[f'WeightedAvg_Property{prop}'] ** 2
            X[f'Property{prop}_sqrt'] = np.sqrt(np.abs(X[f'WeightedAvg_Property{prop}']))
        
        return X

# Load data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Separate features and targets
X = train_data.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
y = train_data[[f'BlendProperty{i}' for i in range(1, 11)]]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Base models for stacking
base_models = [
    ('xgb', XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )),
    ('lgbm', LGBMRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )),
    ('catboost', CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_state=42,
        silent=True
    )),
    ('svr', SVR(
        kernel='rbf',
        C=10,
        gamma='scale'
    )),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    ))
]

# Meta model
meta_model = Ridge(alpha=1.0)

# Create stacking regressor for each target property
models = {}
for prop in range(1, 11):
    print(f"Training model for BlendProperty{prop}")
    
    # Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    # Build full pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(score_func=f_regression, k=100)),
        ('regressor', stacking_regressor)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train[f'BlendProperty{prop}'])
    
    # Validate
    val_pred = pipeline.predict(X_val)
    mape = mean_absolute_percentage_error(y_val[f'BlendProperty{prop}'], val_pred)
    print(f"Validation MAPE for BlendProperty{prop}: {mape:.4f}")
    
    models[f'BlendProperty{prop}'] = pipeline

# Make predictions on test data
predictions = pd.DataFrame()
predictions['ID'] = test_data['ID']

for prop in range(1, 11):
    predictions[f'BlendProperty{prop}'] = models[f'BlendProperty{prop}'].predict(test_data.drop(columns=['ID']))

# Save submission
predictions.to_csv('submission.csv', index=False)
print("Submission file saved as submission.csv")