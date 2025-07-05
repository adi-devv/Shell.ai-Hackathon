import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import optuna
import warnings
from scipy import stats
from scipy.stats import skew, kurtosis
import itertools

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    sample_submission_df = pd.read_csv('dataset/sample_solution.csv')
except FileNotFoundError:
    print("Error: Ensure 'train.csv', 'test.csv', and 'sample_solution.csv' are in 'dataset/'.")
    exit()

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check column names to understand the structure
print("\nColumn names in training data:")
print(train_df.columns.tolist())
print("\nFirst few columns:")
print(train_df.columns[:10].tolist())

# Identify actual column patterns
blend_cols = [col for col in train_df.columns if 'Component' in col and ('vol' in col or col.endswith('_vol') or ('Component' in col and not 'Property' in col and col != 'ID'))]
prop_cols = [col for col in train_df.columns if 'Property' in col and 'Blend' not in col]
target_columns = [col for col in train_df.columns if 'BlendProperty' in col]

print(f"\nIdentified blend columns: {blend_cols[:5]}")
print(f"Identified property columns: {prop_cols[:5]}")
print(f"Identified target columns: {target_columns}")

# If blend columns are not found, try alternative patterns
if not blend_cols:
    # Try finding columns that might be component fractions
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    non_target_numeric = [col for col in numeric_cols if 'BlendProperty' not in col and col != 'ID']
    
    # The first 5 columns (excluding ID) should be blend composition
    if 'ID' in train_df.columns:
        potential_blend_cols = [col for col in train_df.columns if col not in target_columns and col != 'ID'][:5]
    else:
        potential_blend_cols = [col for col in train_df.columns if col not in target_columns][:5]
    
    blend_cols = potential_blend_cols
    print(f"Using alternative blend columns: {blend_cols}")

# Update property columns to exclude blend columns
prop_cols = [col for col in train_df.columns if col not in blend_cols and col not in target_columns and col != 'ID']
print(f"Updated property columns count: {len(prop_cols)}")
print(f"Sample property columns: {prop_cols[:5]}")

# --- 2. Advanced Preprocessing ---
def advanced_preprocessing(train_df, test_df, blend_cols, prop_cols, target_columns):
    """Enhanced preprocessing with multiple techniques"""
    
    # 1. Advanced outlier detection and clipping
    for col in prop_cols:
        # Use IQR method for more robust outlier detection
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip both train and test using training bounds
        train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
        test_df[col] = test_df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 2. Target transformation with Box-Cox where applicable
    target_transformations = {}
    for col in target_columns:
        # Clip targets
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train_df[col] = train_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Apply transformation based on distribution
        if (train_df[col] > 0).all():
            skewness = skew(train_df[col])
            if skewness > 1:
                # Log transformation for highly skewed data
                train_df[f'{col}_log'] = np.log1p(train_df[col])
                target_transformations[col] = 'log'
            elif skewness > 0.5:
                # Square root transformation for moderately skewed data
                train_df[f'{col}_sqrt'] = np.sqrt(train_df[col])
                target_transformations[col] = 'sqrt'
            else:
                target_transformations[col] = 'none'
        else:
            target_transformations[col] = 'none'
    
    # 3. Feature scaling for numerical stability
    scalers = {}
    for col in prop_cols + blend_cols:
        scaler = RobustScaler()
        train_df[f'{col}_scaled'] = scaler.fit_transform(train_df[[col]])
        test_df[f'{col}_scaled'] = scaler.transform(test_df[[col]])
        scalers[col] = scaler
    
    return train_df, test_df, target_transformations, scalers

# Apply advanced preprocessing
train_df, test_df, target_transformations, scalers = advanced_preprocessing(train_df, test_df, blend_cols, prop_cols, target_columns)

# --- 3. Enhanced Feature Engineering ---
def create_advanced_features(df, blend_cols, prop_cols, target_columns, is_train=True):
    """Create comprehensive feature set"""
    
    df_processed = df.drop(columns=['ID'], errors='ignore').copy()
    
    # Start with original features
    original_features = [col for col in df_processed.columns if col not in target_columns + [f'{col}_log' for col in target_columns] + [f'{col}_sqrt' for col in target_columns]]
    df_features = df_processed[original_features].copy()
    
    # Extract component and property numbers from column names
    component_properties = {}
    for col in prop_cols:
        # Try to extract component and property numbers
        parts = col.split('_')
        if len(parts) >= 2:
            try:
                comp_part = parts[0]
                prop_part = parts[1]
                
                # Extract component number
                comp_num = int(''.join(filter(str.isdigit, comp_part)))
                
                # Extract property number
                prop_num = int(''.join(filter(str.isdigit, prop_part)))
                
                if comp_num not in component_properties:
                    component_properties[comp_num] = {}
                component_properties[comp_num][prop_num] = col
            except:
                continue
    
    print(f"Detected component-property structure: {len(component_properties)} components")
    
    # 1. Enhanced Weighted Averages with multiple weighting schemes
    if component_properties:
        max_prop = max([max(props.keys()) for props in component_properties.values()])
        
        for prop_num in range(1, max_prop + 1):
            # Linear weighted average
            weighted_avg_col = f'BlendProperty{prop_num}_WeightedAvg'
            df_features[weighted_avg_col] = 0.0
            
            # Quadratic weighted average
            weighted_avg_quad_col = f'BlendProperty{prop_num}_WeightedAvg_Quad'
            df_features[weighted_avg_quad_col] = 0.0
            
            total_vol_quad = 0
            
            for comp_num, props in component_properties.items():
                if prop_num in props and comp_num <= len(blend_cols):
                    vol_col = blend_cols[comp_num - 1]  # 0-indexed
                    prop_col = props[prop_num]
                    
                    if vol_col in df_processed.columns and prop_col in df_processed.columns:
                        vol_frac = df_processed[vol_col] / 100.0
                        prop_val = df_processed[prop_col]
                        
                        # Linear weighted
                        df_features[weighted_avg_col] += vol_frac * prop_val
                        
                        # Quadratic weighted
                        vol_quad = vol_frac ** 2
                        df_features[weighted_avg_quad_col] += vol_quad * prop_val
                        total_vol_quad += vol_quad
            
            # Normalize quadratic weighted average
            df_features[weighted_avg_quad_col] = df_features[weighted_avg_quad_col] / (total_vol_quad + 1e-8)
    
    # 2. Cross-component interactions
    for i in range(min(3, len(blend_cols))):  # Focus on first 3 components
        for j in range(i + 1, min(3, len(blend_cols))):
            vol1_col = blend_cols[i]
            vol2_col = blend_cols[j]
            
            if vol1_col in df_processed.columns and vol2_col in df_processed.columns:
                # Volume interactions
                df_features[f'{vol1_col}_x_{vol2_col}'] = df_processed[vol1_col] * df_processed[vol2_col]
                df_features[f'{vol1_col}_div_{vol2_col}'] = df_processed[vol1_col] / (df_processed[vol2_col] + 1e-8)
    
    # 3. Statistical aggregations across components for each property
    if component_properties:
        for prop_num in range(1, max_prop + 1):
            props_for_agg = []
            for comp_num, props in component_properties.items():
                if prop_num in props:
                    props_for_agg.append(props[prop_num])
            
            if len(props_for_agg) > 1:
                existing_props = [p for p in props_for_agg if p in df_processed.columns]
                if existing_props:
                    prop_values = df_processed[existing_props]
                    df_features[f'Property{prop_num}_mean'] = prop_values.mean(axis=1)
                    df_features[f'Property{prop_num}_std'] = prop_values.std(axis=1)
                    df_features[f'Property{prop_num}_min'] = prop_values.min(axis=1)
                    df_features[f'Property{prop_num}_max'] = prop_values.max(axis=1)
                    df_features[f'Property{prop_num}_range'] = prop_values.max(axis=1) - prop_values.min(axis=1)
    
    # 4. Dominant component features
    if blend_cols:
        existing_blend_cols = [col for col in blend_cols if col in df_processed.columns]
        
        if existing_blend_cols:
            # Find dominant component
            vol_matrix = df_processed[existing_blend_cols].values
            dominant_idx = np.argmax(vol_matrix, axis=1)
            df_features['dominant_component'] = dominant_idx
            df_features['dominant_vol_fraction'] = np.max(vol_matrix, axis=1) / 100.0
            
            # Concentration measures
            vol_normalized = vol_matrix / 100.0 + 1e-8
            df_features['vol_entropy'] = -np.sum(vol_normalized * np.log(vol_normalized), axis=1)
            df_features['vol_gini'] = 1 - np.sum(vol_normalized**2, axis=1)
    
    # 5. Non-linear transformations for key features
    key_features = []
    if blend_cols:
        key_features.extend([col for col in blend_cols[:2] if col in df_features.columns])
    key_features.extend([col for col in df_features.columns if 'WeightedAvg' in col][:5])
    
    for col in key_features:
        if col in df_features.columns and df_features[col].dtype in ['float64', 'int64']:
            # Polynomial features
            df_features[f'{col}_squared'] = df_features[col] ** 2
            
            # Log and sqrt transformations
            if (df_features[col] > 0).all():
                df_features[f'{col}_log'] = np.log1p(df_features[col])
                df_features[f'{col}_sqrt'] = np.sqrt(df_features[col])
    
    # 6. Target-guided features (only for training)
    if is_train and blend_cols:
        global target_encoding_dict
        target_encoding_dict = {}
        
        for target_col in target_columns:
            if target_col in df_processed.columns:
                # Create bins for continuous variables
                for vol_col in blend_cols[:3]:  # Focus on first 3 components
                    if vol_col in df_processed.columns:
                        try:
                            bins = pd.qcut(df_processed[vol_col], q=15, duplicates='drop')
                            target_encoding_dict[(target_col, vol_col)] = df_processed.groupby(bins)[target_col].mean()
                        except:
                            target_encoding_dict[(target_col, vol_col)] = {}
                        
                        # Map the encoding
                        try:
                            bins_test = pd.cut(df_processed[vol_col], bins=15)
                            df_features[f'{vol_col}_target_mean_{target_col}'] = bins_test.map(target_encoding_dict[(target_col, vol_col)])
                        except:
                            df_features[f'{vol_col}_target_mean_{target_col}'] = 0
    elif not is_train and blend_cols:
        # Apply target encoding to test data
        for target_col in target_columns:
            for vol_col in blend_cols[:3]:
                if vol_col in df_processed.columns and (target_col, vol_col) in target_encoding_dict:
                    try:
                        bins_test = pd.cut(df_processed[vol_col], bins=15)
                        df_features[f'{vol_col}_target_mean_{target_col}'] = bins_test.map(target_encoding_dict[(target_col, vol_col)])
                    except:
                        df_features[f'{vol_col}_target_mean_{target_col}'] = 0
    
    # Fill missing values and handle infinities
    df_features = df_features.fillna(0)
    df_features = df_features.replace([np.inf, -np.inf], 0)
    
    return df_features

# Create enhanced features
print("\nPerforming Advanced Feature Engineering...")
X_train_fe = create_advanced_features(train_df, blend_cols, prop_cols, target_columns, is_train=True)
X_test_fe = create_advanced_features(test_df, blend_cols, prop_cols, target_columns, is_train=False)

# Align columns
train_cols = X_train_fe.columns
test_cols = X_test_fe.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_fe[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_fe[c] = 0
X_test_fe = X_test_fe[train_cols]

print(f"Total features created: {len(train_cols)}")

# --- 4. Advanced Model Selection and Training ---
def create_ensemble_models(X, y, target_idx, n_trials=75):
    """Create optimized ensemble of models"""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['lgb', 'xgb', 'cat'])
        
        if model_type == 'lgb':
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1,
                'random_state': 42
            }
            model = lgb.LGBMRegressor(**params)
            
        elif model_type == 'xgb':
            params = {
                'objective': 'reg:absoluteerror',
                'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42
            }
            model = xgb.XGBRegressor(**params)
            
        else:  # CatBoost
            params = {
                'objective': 'MAE',
                'iterations': trial.suggest_int('iterations', 800, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': 42,
                'verbose': False
            }
            model = CatBoostRegressor(**params)
        
        # Cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            if model_type in ['lgb', 'xgb']:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                         eval_metric='mae', verbose=False)
            else:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
                         use_best_model=True, verbose=False)
            
            pred = model.predict(X_val)
            
            # Transform back if needed
            target_col = target_columns[target_idx]
            if target_transformations.get(target_col) == 'log':
                y_val_orig = np.expm1(y_val)
                pred_orig = np.expm1(pred)
            elif target_transformations.get(target_col) == 'sqrt':
                y_val_orig = y_val ** 2
                pred_orig = pred ** 2
            else:
                y_val_orig = y_val
                pred_orig = pred
            
            mape = mean_absolute_percentage_error(y_val_orig, pred_orig)
            cv_scores.append(mape)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

# Prepare target variables
y_train = pd.DataFrame()

for col in target_columns:
    if target_transformations.get(col) == 'log':
        y_train[col] = train_df[f'{col}_log']
    elif target_transformations.get(col) == 'sqrt':
        y_train[col] = train_df[f'{col}_sqrt']
    else:
        y_train[col] = train_df[col]

# Feature selection per target
print("\nPerforming feature selection...")
feature_sets = {}
for i, col in enumerate(target_columns):
    # Use multiple feature selection methods
    selector1 = SelectKBest(score_func=f_regression, k=min(200, X_train_fe.shape[1]))
    selector1.fit(X_train_fe, y_train[col])
    
    # Random Forest feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_fe, y_train[col])
    
    # Combine selections
    top_features_1 = X_train_fe.columns[selector1.get_support()]
    rf_importance = pd.Series(rf.feature_importances_, index=X_train_fe.columns)
    top_features_2 = rf_importance.nlargest(200).index
    
    # Union of both selections
    combined_features = list(set(top_features_1) | set(top_features_2))
    feature_sets[col] = combined_features[:150]  # Limit to top 150 features
    
    print(f"Selected {len(feature_sets[col])} features for {col}")

# Train models for each target
print("\nTraining models for each target...")
models = {}
test_predictions = {}
oof_predictions = np.zeros((len(X_train_fe), len(target_columns)))

kf = KFold(n_splits=7, shuffle=True, random_state=42)

for i, col in enumerate(target_columns):
    print(f"\nTraining models for {col}...")
    
    # Select features for this target
    X_train_target = X_train_fe[feature_sets[col]]
    X_test_target = X_test_fe[feature_sets[col]]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_target), 
                                  columns=X_train_target.columns, 
                                  index=X_train_target.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_target), 
                                columns=X_test_target.columns, 
                                index=X_test_target.index)
    
    # Optimize model
    best_params = create_ensemble_models(X_train_scaled, y_train[col], i, n_trials=50)
    
    # Train final models with cross-validation
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train[col].iloc[train_idx], y_train[col].iloc[val_idx]
        
        # Create model based on best params
        if best_params['model_type'] == 'lgb':
            model = lgb.LGBMRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae', verbose=False)
        elif best_params['model_type'] == 'xgb':
            model = xgb.XGBRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='mae', verbose=False)
        else:
            model = CatBoostRegressor(**{k: v for k, v in best_params.items() if k != 'model_type'})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], use_best_model=True, verbose=False)
        
        # Predictions
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test_scaled)
        
        oof_predictions[val_idx, i] = val_pred
        test_preds.append(test_pred)
    
    # Average test predictions
    test_predictions[col] = np.mean(test_preds, axis=0)
    
    # Calculate validation MAPE
    target_col = target_columns[i]
    if target_transformations.get(target_col) == 'log':
        y_val_orig = np.expm1(y_train[col])
        pred_orig = np.expm1(oof_predictions[:, i])
    elif target_transformations.get(target_col) == 'sqrt':
        y_val_orig = y_train[col] ** 2
        pred_orig = oof_predictions[:, i] ** 2
    else:
        y_val_orig = y_train[col]
        pred_orig = oof_predictions[:, i]
    
    mape = mean_absolute_percentage_error(y_val_orig, pred_orig)
    print(f"{col} CV MAPE: {mape:.4f}")

# Calculate overall CV MAPE
overall_mapes = []
for i, col in enumerate(target_columns):
    if target_transformations.get(col) == 'log':
        y_val_orig = np.expm1(y_train.iloc[:, i])
        pred_orig = np.expm1(oof_predictions[:, i])
    elif target_transformations.get(col) == 'sqrt':
        y_val_orig = y_train.iloc[:, i] ** 2
        pred_orig = oof_predictions[:, i] ** 2
    else:
        y_val_orig = y_train.iloc[:, i]
        pred_orig = oof_predictions[:, i]
    
    mape = mean_absolute_percentage_error(y_val_orig, pred_orig)
    overall_mapes.append(mape)

overall_cv_mape = np.mean(overall_mapes)
print(f"\nOverall CV MAPE: {overall_cv_mape:.4f}")

# --- 5. Create Final Submission ---
final_predictions = np.zeros((len(X_test_fe), len(target_columns)))

for i, col in enumerate(target_columns):
    pred = test_predictions[col]
    
    # Transform back if needed
    if target_transformations.get(col) == 'log':
        pred = np.expm1(pred)
    elif target_transformations.get(col) == 'sqrt':
        pred = pred ** 2
    
    final_predictions[:, i] = pred

# Create submission dataframe
submission_df = pd.DataFrame({
    'ID': test_df['ID']
})

for i, col in enumerate(target_columns):
    submission_df[col] = final_predictions[:, i]

# Save submission
submission_file = 'enhanced_submission.csv'
submission_df.to_csv(submission_file, index=False)

print(f"\nSubmission saved to: {submission_file}")
print("First 5 rows of submission:")
print(submission_df.head())

# Estimate score
reference_cost_public = 2.72
estimated_score = 100 * max(0, 1 - overall_cv_mape / reference_cost_public)
print(f"\nEstimated Public Leaderboard Score: {estimated_score:.2f}")