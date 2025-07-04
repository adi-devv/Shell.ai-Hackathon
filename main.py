import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import optuna
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    sample_submission_df = pd.read_csv('dataset/sample_solution.csv')
except FileNotFoundError:
    print("Error: Ensure 'train.csv', 'test.csv', and 'sample_solution.csv' are in the 'dataset/' directory.")
    exit()

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# --- 2. Define Features and Targets ---
target_columns = [f'BlendProperty{i}' for i in range(1, 11)]
original_feature_columns = [col for col in train_df.columns if col not in ['ID'] + target_columns]
test_ids = test_df['ID']

# --- 3. Enhanced Feature Engineering ---
def create_features(df, is_train_df=True):
    df_processed = df.drop(columns=['ID'], errors='ignore').copy()
    
    blend_cols = [f'Component{i}_vol' for i in range(1, 6)]
    component_prop_cols_map = {comp_num: [f'Component{comp_num}_Property{prop_num}' for prop_num in range(1, 11)] for comp_num in range(1, 6)}
    
    # Add original features
    features_to_add_back = [col for col in df.columns if col in original_feature_columns]
    df_features = df_processed[features_to_add_back].copy()
    
    # 1. Weighted Averages of Component Properties
    for prop_num in range(1, 11):
        weighted_avg_col = f'BlendProperty{prop_num}_WeightedAvg'
        df_features[weighted_avg_col] = 0.0
        for comp_num in range(1, 6):
            vol_col = f'Component{comp_num}_vol'
            prop_col = f'Component{comp_num}_Property{prop_num}'
            if vol_col in df_processed.columns and prop_col in df_processed.columns:
                df_features[weighted_avg_col] += (df_processed[vol_col] / 100.0) * df_processed[prop_col]
    
    # 2. Polynomial Interactions (Volume x Property, squared terms)
    for comp_num in range(1, 6):
        vol_col = f'Component{comp_num}_vol'
        for prop_num in range(1, 11):
            prop_col = f'Component{comp_num}_Property{prop_num}'
            if vol_col in df_processed.columns and prop_col in df_processed.columns:
                df_features[f'{vol_col}_x_{prop_col}'] = df_processed[vol_col] * df_processed[prop_col]
                df_features[f'{prop_col}_squared'] = df_processed[prop_col] ** 2
    
    # 3. Statistical Aggregations
    for prop_num in range(1, 11):
        props_for_agg = [f'Component{comp_num}_Property{prop_num}' for comp_num in range(1, 6)]
        existing_props_for_agg = [p for p in props_for_agg if p in df_processed.columns]
        if existing_props_for_agg:
            df_features[f'Property{prop_num}_min'] = df_processed[existing_props_for_agg].min(axis=1)
            df_features[f'Property{prop_num}_max'] = df_processed[existing_props_for_agg].max(axis=1)
            df_features[f'Property{prop_num}_mean'] = df_processed[existing_props_for_agg].mean(axis=1)
            df_features[f'Property{prop_num}_std'] = df_processed[existing_props_for_agg].std(axis=1).fillna(0)
            df_features[f'Property{prop_num}_range'] = df_features[f'Property{prop_num}_max'] - df_features[f'Property{prop_num}_min']
    
    # 4. Property Ratios Between Components
    for prop_num in range(1, 11):
        for comp_num1 in range(1, 5):
            for comp_num2 in range(comp_num1 + 1, 6):
                prop_col1 = f'Component{comp_num1}_Property{prop_num}'
                prop_col2 = f'Component{comp_num2}_Property{prop_num}'
                if prop_col1 in df_processed.columns and prop_col2 in df_processed.columns:
                    df_features[f'Prop{prop_num}_Comp{comp_num1}_div_Comp{comp_num2}'] = df_processed[prop_col1] / (df_processed[prop_col2] + 1e-6)
    
    # 5. Log Transformations for Skewed Properties
    for prop_num in range(1, 11):
        props_for_agg = [f'Component{comp_num}_Property{prop_num}' for comp_num in range(1, 6)]
        existing_props_for_agg = [p for p in props_for_agg if p in df_processed.columns]
        for col in existing_props_for_agg:
            df_features[f'{col}_log'] = np.log1p(df_processed[col].clip(lower=0))
    
    return df_features

# Apply feature engineering
print("\nPerforming Feature Engineering...")
X_train_fe = create_features(train_df, is_train_df=True)
X_test_fe = create_features(test_df, is_train_df=False)

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

# Preprocessing: Standardize features
scaler = StandardScaler()
X_train_fe = pd.DataFrame(scaler.fit_transform(X_train_fe), columns=X_train_fe.columns, index=X_train_fe.index)
X_test_fe = pd.DataFrame(scaler.transform(X_test_fe), columns=X_test_fe.columns, index=X_test_fe.index)

y_train = train_df[target_columns]

print("Features after engineering (X_train_fe) shape:", X_train_fe.shape)
print("Features after engineering (X_test_fe) shape:", X_test_fe.shape)

# --- 4. Hyperparameter Tuning with Optuna ---
def objective(trial):
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 31, 128),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    mape_scores = []
    for train_idx, val_idx in kf.split(X_train_fe):
        X_tr, X_val = X_train_fe.iloc[train_idx], X_train_fe.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_val)
        mape = np.mean([mean_absolute_percentage_error(y_val[col], val_preds[:, i]) for i, col in enumerate(target_columns)])
        mape_scores.append(mape)
    return np.mean(mape_scores)

print("\nRunning Optuna for hyperparameter tuning...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# --- 5. Model Training ---
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': best_params['n_estimators'],
    'learning_rate': best_params['learning_rate'],
    'feature_fraction': best_params['feature_fraction'],
    'bagging_fraction': best_params['bagging_fraction'],
    'bagging_freq': 1,
    'lambda_l1': best_params['lambda_l1'],
    'lambda_l2': best_params['lambda_l2'],
    'num_leaves': best_params['num_leaves'],
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}
xgb_params = {
    'objective': 'reg:absoluteerror',
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

# Train LightGBM and XGBoost
lgb_model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))
xgb_model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))

print("\nStarting K-Fold Cross-Validation training...")
kf = KFold(n_splits=7, shuffle=True, random_state=42)  # Increased to 7 folds
oof_preds_lgb = np.zeros(y_train.shape)
oof_preds_xgb = np.zeros(y_train.shape)
test_preds_lgb = []
test_preds_xgb = []
fold_mape_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_fe, y_train)):
    print(f"--- Fold {fold + 1}/{kf.n_splits} ---")
    X_train_fold, X_val_fold = X_train_fe.iloc[train_index], X_train_fe.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Train LightGBM
    lgb_model.fit(X_train_fold, y_train_fold)
    lgb_val_preds = lgb_model.predict(X_val_fold)
    oof_preds_lgb[val_index] = lgb_val_preds
    test_preds_lgb.append(lgb_model.predict(X_test_fe))
    
    # Train XGBoost
    xgb_model.fit(X_train_fold, y_train_fold)
    xgb_val_preds = xgb_model.predict(X_val_fold)
    oof_preds_xgb[val_index] = xgb_val_preds
    test_preds_xgb.append(xgb_model.predict(X_test_fe))
    
    # Blend predictions (weighted average, 0.7 LGB + 0.3 XGB)
    blend_val_preds = 0.7 * lgb_val_preds + 0.3 * xgb_val_preds
    fold_mape = np.mean([mean_absolute_percentage_error(y_val_fold[col], blend_val_preds[:, i]) for i, col in enumerate(target_columns)])
    fold_mape_scores.append(fold_mape)
    print(f"Fold {fold + 1} Average MAPE: {fold_mape:.4f}")

print("\nCross-Validation Complete.")
print("Per-fold Average MAPEs:", [f"{m:.4f}" for m in fold_mape_scores])
overall_cv_mape = np.mean(fold_mape_scores)
print(f"Overall Cross-Validation Average MAPE: {overall_cv_mape:.4f}")

# --- 6. Final Predictions and Submission ---
final_test_predictions_lgb = np.mean(test_preds_lgb, axis=0)
final_test_predictions_xgb = np.mean(test_preds_xgb, axis=0)
final_test_predictions = 0.7 * final_test_predictions_lgb + 0.3 * final_test_predictions_xgb

predictions_df = pd.DataFrame(final_test_predictions, columns=target_columns)
submission_df = pd.DataFrame({'ID': test_ids})
submission_df = pd.concat([submission_df, predictions_df], axis=1)
submission_df = submission_df[['ID'] + target_columns]

submission_file_name = 'submission_improved.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())

# Estimate leaderboard score
reference_cost_public = 2.72
estimated_score_public = 100 * max(0, 1 - overall_cv_mape / reference_cost_public)
print(f"\nEstimated Public Leaderboard Score: {estimated_score_public:.2f}")