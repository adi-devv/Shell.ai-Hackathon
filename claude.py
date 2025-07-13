import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import optuna
import warnings
warnings.filterwarnings('ignore')

# Enhanced Feature Engineering with domain knowledge
class EnhancedFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.kmeans_models = {}
        self.pca_models = {}
    
    def fit(self, X, y=None):
        # Fit clustering models for each property group
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        for prop in range(1, 11):
            prop_cols = [f'{comp}_Property{prop}' for comp in components]
            if all(col in X.columns for col in prop_cols):
                # Fit KMeans for property clustering
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                kmeans.fit(X[prop_cols])
                self.kmeans_models[prop] = kmeans
                
                # Fit PCA for dimensionality reduction
                pca = PCA(n_components=min(3, len(prop_cols)))
                pca.fit(X[prop_cols])
                self.pca_models[prop] = pca
        
        return self
    
    def transform(self, X):
        X = X.copy()
        components = ['Component1', 'Component2', 'Component3', 'Component4', 'Component5']
        
        # 1. Enhanced weighted averages with different weighting schemes
        for prop in range(1, 11):
            weighted_props = []
            prop_cols = [f'{comp}_Property{prop}' for comp in components]
            frac_cols = [f'{comp}_fraction' for comp in components]
            
            if all(col in X.columns for col in prop_cols + frac_cols):
                # Linear weighted average
                for comp, frac in zip(components, frac_cols):
                    weighted_props.append(X[frac] * X[f'{comp}_Property{prop}'])
                X[f'WeightedAvg_Property{prop}'] = sum(weighted_props)
                
                # Quadratic weighted average (gives more weight to dominant components)
                quad_weighted = []
                for comp, frac in zip(components, frac_cols):
                    quad_weighted.append(X[frac]**2 * X[f'{comp}_Property{prop}'])
                X[f'QuadWeightedAvg_Property{prop}'] = sum(quad_weighted)
                
                # Entropy-based weighting
                fractions = np.array([X[frac] for frac in frac_cols]).T
                fractions = fractions + 1e-10  # Avoid log(0)
                entropy = -np.sum(fractions * np.log(fractions), axis=1)
                X[f'MixingEntropy_Property{prop}'] = entropy
                
                # Statistical features
                props = np.array([X[f'{comp}_Property{prop}'] for comp in components]).T
                X[f'StdDev_Property{prop}'] = np.std(props, axis=1)
                X[f'Range_Property{prop}'] = np.max(props, axis=1) - np.min(props, axis=1)
                X[f'Skew_Property{prop}'] = pd.DataFrame(props).skew(axis=1)
                X[f'Kurt_Property{prop}'] = pd.DataFrame(props).kurtosis(axis=1)
                
                # Clustering features
                if prop in self.kmeans_models:
                    cluster_labels = self.kmeans_models[prop].predict(X[prop_cols])
                    X[f'Property{prop}_Cluster'] = cluster_labels
                    
                    # Distance to cluster centers
                    distances = self.kmeans_models[prop].transform(X[prop_cols])
                    X[f'Property{prop}_MinClusterDist'] = np.min(distances, axis=1)
                
                # PCA features
                if prop in self.pca_models:
                    pca_features = self.pca_models[prop].transform(X[prop_cols])
                    for i in range(pca_features.shape[1]):
                        X[f'Property{prop}_PCA{i+1}'] = pca_features[:, i]
        
        # 2. Advanced interaction features
        # Component fraction interactions
        for i in range(1, 6):
            for j in range(i+1, 6):
                comp_i = f'Component{i}_fraction'
                comp_j = f'Component{j}_fraction'
                if comp_i in X.columns and comp_j in X.columns:
                    X[f'Comp{i}_Comp{j}_interaction'] = X[comp_i] * X[comp_j]
                    X[f'Comp{i}_Comp{j}_ratio'] = X[comp_i] / (X[comp_j] + 1e-10)
        
        # Property cross-interactions (focus on most important properties)
        important_props = [1, 2, 3, 5, 7, 8]  # Based on typical chemical importance
        for prop1 in important_props:
            for prop2 in important_props:
                if prop1 < prop2:
                    col1 = f'WeightedAvg_Property{prop1}'
                    col2 = f'WeightedAvg_Property{prop2}'
                    if col1 in X.columns and col2 in X.columns:
                        X[f'Prop{prop1}_Prop{prop2}_interaction'] = X[col1] * X[col2]
                        X[f'Prop{prop1}_Prop{prop2}_ratio'] = X[col1] / (X[col2] + 1e-10)
        
        # 3. Polynomial and transformation features
        for prop in important_props:
            col = f'WeightedAvg_Property{prop}'
            if col in X.columns:
                X[f'Property{prop}_squared'] = X[col] ** 2
                X[f'Property{prop}_cubed'] = X[col] ** 3
                X[f'Property{prop}_sqrt'] = np.sqrt(np.abs(X[col]))
                X[f'Property{prop}_log'] = np.log(np.abs(X[col]) + 1e-10)
                X[f'Property{prop}_exp'] = np.exp(np.clip(X[col], -10, 10))
        
        # 4. Compositional features
        frac_cols = [f'Component{i}_fraction' for i in range(1, 6)]
        if all(col in X.columns for col in frac_cols):
            # Diversity indices
            fractions = np.array([X[col] for col in frac_cols]).T
            fractions = fractions + 1e-10
            
            # Shannon diversity
            X['Shannon_Diversity'] = -np.sum(fractions * np.log(fractions), axis=1)
            
            # Simpson diversity
            X['Simpson_Diversity'] = 1 - np.sum(fractions ** 2, axis=1)
            
            # Effective number of components
            X['Effective_Components'] = np.exp(X['Shannon_Diversity'])
            
            # Dominance (fraction of most abundant component)
            X['Dominance'] = np.max(fractions, axis=1)
            
            # Evenness
            X['Evenness'] = X['Shannon_Diversity'] / np.log(5)
        
        return X

# Hyperparameter optimization function
def optimize_model(X_train, y_train, model_type='xgb', n_trials=100):
    def objective(trial):
        if model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            model = XGBRegressor(**params)
        elif model_type == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            model = LGBMRegressor(**params)
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': 42,
                'silent': True
            }
            model = CatBoostRegressor(**params)
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

# Load data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Separate features and targets
X = train_data.drop(columns=[f'BlendProperty{i}' for i in range(1, 11)])
y = train_data[[f'BlendProperty{i}' for i in range(1, 11)]]

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Enhanced preprocessing pipeline
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Create models dictionary with optimized hyperparameters
models = {}
overall_predictions = []

for prop in range(1, 11):
    print(f"\n=== Training models for BlendProperty{prop} ===")
    
    # Prepare data
    y_current = y_train[f'BlendProperty{prop}']
    
    # Feature engineering
    feature_engineer = EnhancedFeatureEngineer()
    X_train_eng = feature_engineer.fit_transform(X_train)
    
    # Preprocessing
    X_train_processed = preprocessor.fit_transform(X_train_eng)
    
    # Feature selection with multiple methods
    # Method 1: Statistical selection
    selector_stats = SelectKBest(score_func=f_regression, k=min(150, X_train_processed.shape[1]))
    X_train_selected = selector_stats.fit_transform(X_train_processed, y_current)
    
    print(f"Features after selection: {X_train_selected.shape[1]}")
    
    # Optimize key models (reduced trials for faster execution)
    print("Optimizing XGBoost...")
    xgb_params = optimize_model(X_train_selected, y_current, 'xgb', n_trials=50)
    
    print("Optimizing LightGBM...")
    lgbm_params = optimize_model(X_train_selected, y_current, 'lgbm', n_trials=50)
    
    print("Optimizing CatBoost...")
    catboost_params = optimize_model(X_train_selected, y_current, 'catboost', n_trials=50)
    
    # Define optimized base models
    base_models = [
        ('xgb', XGBRegressor(**xgb_params)),
        ('lgbm', LGBMRegressor(**lgbm_params)),
        ('catboost', CatBoostRegressor(**catboost_params)),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)),
        ('ridge', Ridge(alpha=10.0)),
        ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000)),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        ))
    ]
    
    # Multiple ensemble strategies
    # Strategy 1: Stacking
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=7,  # Increased CV folds
        n_jobs=-1
    )
    
    # Strategy 2: Voting
    voting_regressor = VotingRegressor(
        estimators=base_models,
        n_jobs=-1
    )
    
    # Train both ensembles
    stacking_regressor.fit(X_train_selected, y_current)
    voting_regressor.fit(X_train_selected, y_current)
    
    # Validation
    X_val_eng = feature_engineer.transform(X_val)
    X_val_processed = preprocessor.transform(X_val_eng)
    X_val_selected = selector_stats.transform(X_val_processed)
    
    stacking_pred = stacking_regressor.predict(X_val_selected)
    voting_pred = voting_regressor.predict(X_val_selected)
    
    # Blend predictions from both ensembles
    blended_pred = 0.6 * stacking_pred + 0.4 * voting_pred
    
    # Calculate metrics
    stacking_mape = mean_absolute_percentage_error(y_val[f'BlendProperty{prop}'], stacking_pred)
    voting_mape = mean_absolute_percentage_error(y_val[f'BlendProperty{prop}'], voting_pred)
    blended_mape = mean_absolute_percentage_error(y_val[f'BlendProperty{prop}'], blended_pred)
    
    print(f"Stacking MAPE: {stacking_mape:.4f}")
    print(f"Voting MAPE: {voting_mape:.4f}")
    print(f"Blended MAPE: {blended_mape:.4f}")
    
    # Store the best performing model combination
    models[f'BlendProperty{prop}'] = {
        'feature_engineer': feature_engineer,
        'preprocessor': preprocessor,
        'selector': selector_stats,
        'stacking': stacking_regressor,
        'voting': voting_regressor,
        'blend_weights': (0.6, 0.4)  # Can be optimized further
    }

# Generate predictions on test data
print("\n=== Generating test predictions ===")
predictions = pd.DataFrame()
predictions['ID'] = test_data['ID']

for prop in range(1, 11):
    print(f"Predicting BlendProperty{prop}...")
    
    model_dict = models[f'BlendProperty{prop}']
    
    # Process test data
    X_test_eng = model_dict['feature_engineer'].transform(test_data.drop(columns=['ID']))
    X_test_processed = model_dict['preprocessor'].transform(X_test_eng)
    X_test_selected = model_dict['selector'].transform(X_test_processed)
    
    # Get predictions from both ensembles
    stacking_pred = model_dict['stacking'].predict(X_test_selected)
    voting_pred = model_dict['voting'].predict(X_test_selected)
    
    # Blend predictions
    w1, w2 = model_dict['blend_weights']
    blended_pred = w1 * stacking_pred + w2 * voting_pred
    
    predictions[f'BlendProperty{prop}'] = blended_pred

# Save submission
predictions.to_csv('enhanced_submission.csv', index=False)
print("\nEnhanced submission file saved as enhanced_submission.csv")
print("Key improvements:")
print("1. Advanced feature engineering with domain knowledge")
print("2. Hyperparameter optimization using Optuna")
print("3. Multiple ensemble strategies (Stacking + Voting)")
print("4. Enhanced preprocessing with RobustScaler")
print("5. Blended predictions from multiple models")
print("6. Improved feature selection methods")
print("7. More sophisticated base models")