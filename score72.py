import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
import warnings

# Suppress potential warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    # Assuming 'dataset/' prefix is correct based on your code
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    sample_submission_df = pd.read_csv('dataset/sample_solution.csv')  # Corrected name from sample_submission.csv
except FileNotFoundError:
    print("Error: Make sure 'train.csv', 'test.csv', and 'sample_solution.csv' are in the 'dataset/' directory.")
    exit()

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# --- 2. Define Features and Targets (Moved before create_features call) ---
# Identify target columns
target_columns = [f'BlendProperty{i}' for i in range(1, 11)]

# Identify original feature columns (Blend Composition + Component Properties)
# All columns except 'ID' and the 10 target properties
original_feature_columns = [col for col in train_df.columns if col not in ['ID'] + target_columns]

# Store test IDs for submission
test_ids = test_df['ID']

print("\nOriginal Features for training shape (before FE):", train_df[original_feature_columns].shape)
print("Targets for training shape:", train_df[target_columns].shape)
print("Original Features for testing shape (before FE):", test_df[original_feature_columns].shape)


# --- 3. Feature Engineering ---

def create_features(df, is_train_df=True):
    # Ensure ID is handled if present, typically dropped before FE
    df_processed = df.drop(columns=['ID'], errors='ignore').copy()

    # Get blend composition columns (these are actual features)
    blend_cols = [f'Component{i}_vol' for i in range(1, 6)]

    # Get component property columns
    component_prop_cols_map = {}
    for comp_num in range(1, 6):
        component_prop_cols_map[comp_num] = [f'Component{comp_num}_Property{prop_num}' for prop_num in range(1, 11)]

    # Add original features to the processed dataframe explicitly, ensures consistency
    # Only add columns that are actual features (not targets in train_df)
    features_to_add_back = [col for col in df.columns if col in original_feature_columns]
    df_features = df_processed[features_to_add_back].copy()

    # 1. Weighted Averages of Component Properties (Weighted by Volume)
    for prop_num in range(1, 11):
        weighted_avg_col = f'BlendProperty{prop_num}_WeightedAvg'
        df_features[weighted_avg_col] = 0.0  # Initialize as float
        for comp_num in range(1, 6):
            vol_col = f'Component{comp_num}_vol'
            prop_col = f'Component{comp_num}_Property{prop_num}'
            if vol_col in df_processed.columns and prop_col in df_processed.columns:
                df_features[weighted_avg_col] += (df_processed[vol_col] / 100.0) * df_processed[prop_col]

    # 2. Interactions between Blend Volume and Component Properties
    for comp_num in range(1, 6):
        vol_col = f'Component{comp_num}_vol'
        for prop_num in range(1, 11):
            prop_col = f'Component{comp_num}_Property{prop_num}'
            if vol_col in df_processed.columns and prop_col in df_processed.columns:
                df_features[f'{vol_col}_x_{prop_col}'] = df_processed[vol_col] * df_processed[prop_col]

    # 3. Statistical Aggregations Across Component Properties
    # (e.g., min, max, mean, std of Property1 across all components)
    for prop_num in range(1, 11):
        props_for_agg = [f'Component{comp_num}_Property{prop_num}' for comp_num in range(1, 6)]
        # Filter props_for_agg to ensure columns actually exist in df_processed
        existing_props_for_agg = [p for p in props_for_agg if p in df_processed.columns]

        if existing_props_for_agg:  # Only proceed if there are columns to aggregate
            df_features[f'Property{prop_num}_min'] = df_processed[existing_props_for_agg].min(axis=1)
            df_features[f'Property{prop_num}_max'] = df_processed[existing_props_for_agg].max(axis=1)
            df_features[f'Property{prop_num}_mean'] = df_processed[existing_props_for_agg].mean(axis=1)
            df_features[f'Property{prop_num}_std'] = df_processed[existing_props_for_agg].std(axis=1).fillna(
                0)  # std can be NaN if only one component

    # 4. Differences between specific component properties (example for Property1)
    # This is more exploratory; you'd need domain knowledge or try many combinations
    # For now, let's do a couple of simple ones if there are at least two components
    if len(blend_cols) >= 2:  # Check if there are at least two components for difference
        for prop_num in range(1, 11):
            comp1_prop = f'Component1_Property{prop_num}'
            comp2_prop = f'Component2_Property{prop_num}'
            if comp1_prop in df_processed.columns and comp2_prop in df_processed.columns:
                df_features[f'Component1_Prop{prop_num}_minus_Component2_Prop{prop_num}'] = \
                    df_processed[comp1_prop] - df_processed[comp2_prop]
            # Add more specific differences/ratios if hypotheses exist, e.g., for specific property types

    return df_features


print("\nPerforming Feature Engineering...")
# Pass the correct DataFrame to create_features
X_train_fe = create_features(train_df, is_train_df=True)
X_test_fe = create_features(test_df, is_train_df=False)

# Align columns - crucial after feature engineering, especially if some features are only present in train or test
# due to sparse data or feature generation quirks. This ensures both sets have identical columns in order.
train_cols = X_train_fe.columns
test_cols = X_test_fe.columns

missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test_fe[c] = 0  # Add missing columns to test set, fill with 0 or appropriate default

missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train_fe[c] = 0  # Add missing columns to train set

# Ensure the order of columns is the same for both train and test
X_test_fe = X_test_fe[train_cols]

# The y_train is already defined correctly as it only uses target_columns from the original train_df
y_train = train_df[target_columns]

print("Features after engineering (X_train_fe) shape:", X_train_fe.shape)
print("Features after engineering (X_test_fe) shape:", X_test_fe.shape)

# --- 4. Model Training (Using LightGBM with Cross-Validation) ---

# Define the base LightGBM regressor
# Hyperparameters are crucial for GBMs. These are a starting point.
# You'd typically tune these with GridSearchCV or RandomizedSearchCV.
lgb_params = {
    'objective': 'regression_l1',  # MAE objective, suitable for MAPE-like metrics
    'metric': 'mae',
    'n_estimators': 1500,  # Increased estimators
    'learning_rate': 0.01,  # Smaller learning rate
    'feature_fraction': 0.8,  # Fraction of features considered per tree
    'bagging_fraction': 0.8,  # Fraction of data used per tree (sampling)
    'bagging_freq': 1,
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1,  # L2 regularization
    'num_leaves': 64,  # Increased complexity
    'verbose': -1,  # Suppress verbose output
    'n_jobs': -1,  # Use all available cores
    'seed': 42,
    'boosting_type': 'gbdt',  # Traditional Gradient Boosting Decision Tree
}

# Wrap LightGBM in MultiOutputRegressor
model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))

print("\nStarting K-Fold Cross-Validation training...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation

oof_preds = np.zeros(y_train.shape)  # Out-Of-Fold predictions for local validation
test_preds_folds = []  # Store predictions for test set from each fold

fold_mape_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_train_fe, y_train)):
    print(f"--- Fold {fold + 1}/{kf.n_splits} ---")
    X_train_fold, X_val_fold = X_train_fe.iloc[train_index], X_train_fe.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Train the model for the current fold
    model.fit(X_train_fold, y_train_fold)

    # Make OOF predictions
    fold_val_preds = model.predict(X_val_fold)
    oof_preds[val_index] = fold_val_preds

    # Make predictions on the test set for this fold
    fold_test_preds = model.predict(X_test_fe)
    test_preds_folds.append(fold_test_preds)

    # Evaluate MAPE for this fold
    fold_mape = []
    for i, target_col in enumerate(target_columns):
        mape_val = mean_absolute_percentage_error(y_val_fold[target_col], fold_val_preds[:, i])
        fold_mape.append(mape_val)
    avg_fold_mape = np.mean(fold_mape)
    fold_mape_scores.append(avg_fold_mape)
    print(f"Fold {fold + 1} Average MAPE: {avg_fold_mape:.4f}")

print("\nCross-Validation Complete.")
print("Per-fold Average MAPEs:", [f"{m:.4f}" for m in fold_mape_scores])
overall_cv_mape = np.mean(fold_mape_scores)
print(f"Overall Cross-Validation Average MAPE: {overall_cv_mape:.4f}")

# --- 5. Final Predictions and Submission ---

# Average predictions from all folds for the test set (Ensembling)
final_test_predictions = np.mean(test_preds_folds, axis=0)

# Ensure predictions are in a DataFrame with correct column names
predictions_df = pd.DataFrame(final_test_predictions, columns=target_columns)

# --- 6. Generate Submission File ---
submission_df = pd.DataFrame({'ID': test_ids})
submission_df = pd.concat([submission_df, predictions_df], axis=1)

# Ensure the column order is correct as per submission guidelines
submission_df = submission_df[['ID'] + target_columns]

# Save the submission file
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"\nSubmission file '{submission_file_name}' created successfully!")
print("First 5 rows of the submission file:")
print(submission_df.head())

# Calculate estimated public leaderboard score based on CV MAPE
reference_cost_public = 2.72  # For public leaderboard
estimated_score_public = 100 * max(0, 1 - overall_cv_mape / reference_cost_public)
print(f"\nEstimated Public Leaderboard Score (based on CV MAPE): {estimated_score_public:.2f}")