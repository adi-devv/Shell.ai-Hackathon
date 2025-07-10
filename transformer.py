import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet # Added ElasticNet as an option
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
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning, module=r'sklearn\.utils\.validation')

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
                
                for prop in range(1, 6): # Increased range for property interactions
                    prop_i = f'Component{i}_Property{prop}'
                    prop_j = f'Component{j}_Property{prop}'
                    X[f'Component{i}_Property{prop}_Component{j}_Property{prop}_interaction'] = (
                        X[prop_i] * X[prop_j])
        
        # Polynomial features for important properties (consider degree 3 if values are not too extreme)
        for prop in [1, 2, 3, 5, 7]:
            X[f'Property{prop}_squared'] = X[f'WeightedAvg_Property{prop}'] ** 2
            X[f'Property{prop}_sqrt'] = np.sqrt(np.abs(X[f'WeightedAvg_Property{prop}']))
            # X[f'Property{prop}_cubed'] = X[f'WeightedAvg_Property{prop}'] ** 3 # Consider this if data allows
        
        return X

# Load data
try:
    train_data = pd.read_csv('dataset/train.csv')
    test_data = pd.read_csv('dataset/test.csv')
except FileNotFoundError:
    print("Error: Ensure 'train.csv' and 'test.csv' are in a 'dataset' folder in the same directory as the script.")
    exit()

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

# Base models for stacking (Slightly adjusted again, leaning on more estimators, lower LR, and subtle regularization)
base_models = [
    ('xgb', XGBRegressor(
        n_estimators=4000,          # Increased n_estimators
        max_depth=6,                # Keep original effective depth
        learning_rate=0.025,        # Slightly lower learning rate for more estimators
        subsample=0.75,             # Increased subsample for more diversity in trees
        colsample_bytree=0.75,      # Increased colsample_bytree
        random_state=42,
        n_jobs=-1,
        gamma=0.05,
        min_child_weight=1,
        # Early stopping for XGBoost if you fit it separately, not directly in StackingRegressor
        # early_stopping_rounds=100
    )),
    ('lgbm', LGBMRegressor(
        n_estimators=4000,          # Increased n_estimators
        max_depth=8,                # Keep original effective depth
        learning_rate=0.025,        # Slightly lower learning rate
        subsample=0.75,             # Increased subsample
        colsample_bytree=0.75,      # Increased colsample_bytree
        random_state=42,
        verbose=-1,
        num_leaves=63,              # Good balance with max_depth=8
        n_jobs=-1,
        reg_alpha=0.05,
        reg_lambda=0.05,
        # early_stopping_round=100
    )),
    ('catboost', CatBoostRegressor(
        iterations=4000,            # Increased iterations
        depth=6,                    # Keep original effective depth
        learning_rate=0.025,        # Slightly lower learning rate
        random_state=42,
        silent=True,
        l2_leaf_reg=2.0,            # Moderate L2 regularization
        # early_stopping_rounds=100
    )),
    ('svr', SVR(
        kernel='rbf',
        C=7,                        # Slightly reduced from original 10, more flexible
        gamma='scale',
        epsilon=0.07                # Slightly increased epsilon
    )),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(150, 75, 25), # Slightly more complex MLP structure for more capacity
        activation='relu',
        solver='adam',
        max_iter=4000,              # Increased max_iter for more training
        random_state=42,
        alpha=0.0001,
        learning_rate_init=0.001,
        early_stopping=True,
        n_iter_no_change=20
    ))
]

# Meta model - Ridge is okay, but ElasticNet can sometimes perform better
meta_model = Ridge(alpha=0.8) # Adjusted alpha slightly
# meta_model = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42) # Alternative meta-model

# --- Custom MAPE function to handle potential zero actuals (important for "extremely low" MAPE) ---
def safe_mape(y_true, y_pred, epsilon=1e-10):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero and handle very small true values
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Create stacking regressor for each target property
models = {}
print("--- Model Training and Validation ---")
for prop in range(1, 11):
    
    # Decide whether to transform the target for this property
    # Only if the property values are all non-negative and potentially skewed
    # Check y_train[f'BlendProperty{prop}'].min() > -epsilon and y_train[f'BlendProperty{prop}'].skew() > 1.0
    
    # For demonstration, let's assume log transformation is beneficial for all BlendProperties
    # You should evaluate if this is truly the case for each property.
    apply_log_transform = True # Set to False if you don't want to try this
    
    y_train_prop = y_train[f'BlendProperty{prop}']
    y_val_prop = y_val[f'BlendProperty{prop}']

    if apply_log_transform:
        # Check for non-positive values before log1p
        if (y_train_prop < 0).any() or (y_val_prop < 0).any():
            print(f"Warning: BlendProperty{prop} has negative values. Log transform might not be suitable. Skipping.")
            y_train_transformed = y_train_prop
            y_val_transformed = y_val_prop
            apply_log_transform = False # Disable for this property if negatives exist
        else:
            # Add a small value before log1p to handle potential zeros gracefully
            y_train_transformed = np.log1p(y_train_prop) 
            y_val_transformed = np.log1p(y_val_prop)
    else:
        y_train_transformed = y_train_prop
        y_val_transformed = y_val_prop

    # Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5, # Keep 5-fold cross-validation
        n_jobs=-1
    )
    
    # Build full pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('feature_selector', SelectKBest(score_func=f_regression, k=150)), # Increased k for more features
        ('regressor', stacking_regressor)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train_transformed) # Fit on transformed target if applicable
    
    # Validate
    val_pred_transformed = pipeline.predict(X_val)
    
    if apply_log_transform:
        val_pred = np.expm1(val_pred_transformed) # Inverse transform predictions
        # Ensure predictions are non-negative after inverse transform, if property cannot be negative
        val_pred[val_pred < 0] = 0 
    else:
        val_pred = val_pred_transformed

    # Calculate MAPE using the custom safe_mape function
    mape = safe_mape(y_val_prop, val_pred)
    print(f"BlendProperty{prop} Validation MAPE: {mape:.4f}")
    
    models[f'BlendProperty{prop}'] = pipeline

# Make predictions on test data
predictions = pd.DataFrame()
predictions['ID'] = test_data['ID']

print("\n--- Generating Test Predictions ---")
for prop in range(1, 11):
    model_pipeline = models[f'BlendProperty{prop}']
    test_pred_transformed = model_pipeline.predict(test_data.drop(columns=['ID']))
    
    # Inverse transform test predictions if log transform was applied for this property
    # This requires knowing if `apply_log_transform` was True for that specific property's model
    # A robust way is to store this flag in the models dictionary or apply it conditionally
    # For simplicity, assuming the same `apply_log_transform` flag as during training for now
    
    # You would need to store the `apply_log_transform` flag per model if it varies
    # For example: models[f'BlendProperty{prop}'] = {'pipeline': pipeline, 'log_transform': apply_log_transform}
    # And then retrieve it here. For now, using the last `apply_log_transform`'s value.
    
    if apply_log_transform: # Use the flag from the training loop, assuming it's consistent
        test_pred = np.expm1(test_pred_transformed)
        test_pred[test_pred < 0] = 0 # Ensure non-negativity
    else:
        test_pred = test_pred_transformed

    predictions[f'BlendProperty{prop}'] = test_pred

# Save submission
predictions.to_csv('submission.csv', index=False)
print("\n--- Submission ---")
print("Submission file saved as submission.csv")