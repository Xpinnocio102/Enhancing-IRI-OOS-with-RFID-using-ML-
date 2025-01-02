
# -----------------------------------------------------------------------------------
# 0. Install Necessary Libraries
# -----------------------------------------------------------------------------------

# Note: In a Jupyter notebook, remove the '!' in front of pip install commands
# Uncomment the following lines if running in a Jupyter environment

!pip install pandas numpy scikit-learn matplotlib seaborn optuna xgboost lightgbm catboost shap joblib imbalanced-learn featuretools

# -----------------------------------------------------------------------------------
# 1. Import Necessary Libraries
# -----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report,
                             mean_squared_error, mean_absolute_error, r2_score, roc_curve,
                             precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# For Jupyter notebooks, uncomment the following line
# %matplotlib inline

# -----------------------------------------------------------------------------------
# 2. Set Random State for Reproducibility
# -----------------------------------------------------------------------------------
RANDOM_STATE = 42

# -----------------------------------------------------------------------------------
# 3. Check GPU Availability (for XGBoost and CatBoost)
# -----------------------------------------------------------------------------------
import subprocess

GPU_AVAILABLE = False
try:
    result = subprocess.check_output(['nvidia-smi'])
    GPU_AVAILABLE = True
    logging.info("GPU detected. Models will utilize GPU acceleration where applicable.")
except:
    GPU_AVAILABLE = False
    logging.info("No GPU detected. Models will utilize CPU.")

# -----------------------------------------------------------------------------------
# 4. Load the Dataset
# -----------------------------------------------------------------------------------
try:
    data = pd.read_csv('/content/ZARA_RFID_File_Final.csv')
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error("Error: 'ZARA_RFID_File_Final.csv' not found. Please check the file path.")
    raise
except pd.errors.ParserError:
    logging.error("Error: Could not parse 'ZARA_RFID_File_Final.csv'. Please check the file format and delimiter.")
    raise
except Exception as e:
    logging.error(f"An unexpected error occurred while loading the dataset: {e}")
    raise

# -----------------------------------------------------------------------------------
# 5. Initial Data Exploration and Preprocessing (EDA)
# -----------------------------------------------------------------------------------
print("Initial Data Snapshot:")
print(data.head())

# Check for target variable 'will_stockout'
if 'will_stockout' in data.columns:
    print("\nTarget variable 'will_stockout' is present in the dataset.")
    # Encode 'will_stockout' as binary: YES=1, NO=0
    data['will_stockout'] = data['will_stockout'].map({'YES': 1, 'NO': 0})
    print("Unique values after encoding:", data['will_stockout'].unique())
else:
    print("\nTarget variable 'will_stockout' is missing from the dataset.")
    # Depending on your pipeline requirements, you might want to handle this differently

# Parse 'scraped_at' to datetime
if 'scraped_at' in data.columns:
    data['scraped_at'] = pd.to_datetime(data['scraped_at'], errors='coerce')
    # Fill any missing datetime values by forward filling
    data['scraped_at'] = data['scraped_at'].fillna(method='ffill')
    print("\n'scraped_at' after parsing and filling missing values:")
    print(data['scraped_at'].head())
else:
    logging.warning("'scraped_at' column not found in the dataset.")

# -----------------------------------------------------------------------------------
# 6. Encode Categorical Variables
# -----------------------------------------------------------------------------------
# Encode binary categorical variables
binary_columns = ['is_discounted', 'is_returnable']
for col in binary_columns:
    if col in data.columns:
        data[col] = data[col].map({'Yes': 1, 'No': 0})
    else:
        logging.warning(f"Column '{col}' not found in the dataset.")

print("\nEncoded Binary Columns:")
print(data[binary_columns].head() if binary_columns else "No binary columns to display.")

# One-hot encode 'gender' (assuming only two categories: Man, Woman)
if 'gender' in data.columns:
    print("\nUnique values in 'gender':", data['gender'].unique())
    data['gender'] = data['gender'].str.capitalize()
    data = pd.get_dummies(data, columns=['gender'], drop_first=True)
    print("\nEncoded 'gender' Column:")
    print(data.filter(like='gender_').head())
else:
    logging.warning("Column 'gender' not found in the dataset.")

# One-hot encode other categorical variables
categorical_columns = ['location_type', 'subcategory', 'category']
for col in categorical_columns:
    if col in data.columns:
        data = pd.get_dummies(data, columns=[col], prefix=col, drop_first=False)
    else:
        logging.warning(f"Column '{col}' not found in the dataset.")

print("\nEncoded Categorical Columns:")
encoded_cols = [col for col in data.columns if col.startswith('location_type_') or col.startswith('subcategory_') or col.startswith('category_')]
print(data[encoded_cols].head() if encoded_cols else "No additional categorical columns to display.")

# Encode 'RFID_Tag'
if 'RFID_Tag' in data.columns:
    data['RFID_Tag'] = data['RFID_Tag'].map({'YES': 1, 'NO': 0})
    print("\nEncoded 'RFID_Tag' Column:")
    print(data['RFID_Tag'].head())
else:
    logging.warning("Column 'RFID_Tag' not found in the dataset.")

# -----------------------------------------------------------------------------------
# 7. Handle Missing Values
# -----------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer

# Impute missing numerical values with mean
numerical_columns = ['price', 'stock_level', 'actual_stock']
imputer = SimpleImputer(strategy='mean')
for col in numerical_columns:
    if col in data.columns:
        data[col] = imputer.fit_transform(data[[col]])
    else:
        logging.warning(f"Numerical column '{col}' not found in the dataset.")

print("\nNumeric Columns After Imputation:")
print(data[numerical_columns].isnull().sum())

# -----------------------------------------------------------------------------------
# 8. Advanced Feature Engineering
# -----------------------------------------------------------------------------------
print("\n--- Advanced Feature Engineering ---")

# Create new features based on existing ones
if {'price', 'stock_level'}.issubset(data.columns):
    data['price_log'] = np.log1p(data['price'])
    data['stock_price_ratio'] = data['stock_level'] / data['price'].replace(0, np.nan)
    data['price_stock_interaction'] = data['price'] * data['stock_level']
    data['hour_of_day'] = data['scraped_at'].dt.hour if 'scraped_at' in data.columns else 0
    data['day_of_week'] = data['scraped_at'].dt.dayofweek if 'scraped_at' in data.columns else 0
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    data['price_squared'] = data['price'] ** 2
    data['stock_level_squared'] = data['stock_level'] ** 2

    # Fill any NaN values in 'stock_price_ratio' with 0 (after division by zero)
    data['stock_price_ratio'] = data['stock_price_ratio'].fillna(0)

    print("\nEngineered Features:")
    engineered_features = ['price_log', 'stock_price_ratio', 'price_stock_interaction',
                           'hour_of_day', 'day_of_week', 'is_weekend',
                           'price_squared', 'stock_level_squared']
    print(data[engineered_features].head())
else:
    logging.warning("Required columns for feature engineering ('price', 'stock_level') not found.")

# -----------------------------------------------------------------------------------
# 9. Text Feature Extraction
# -----------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

print("\n--- Text Feature Extraction ---")
# Extract features from 'description' using TF-IDF and SVD
if 'description' in data.columns:
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['description'].astype(str))
    svd = TruncatedSVD(n_components=20, random_state=RANDOM_STATE)
    tfidf_svd = svd.fit_transform(tfidf_matrix)
    tfidf_svd_df = pd.DataFrame(tfidf_svd, columns=[f'tfidf_svd_{i}' for i in range(20)], index=data.index)
    data = pd.concat([data, tfidf_svd_df], axis=1)
    print("\nText Features Extracted and Added to Dataset:")
    print(tfidf_svd_df.head())
else:
    print("\n'description' column not found. Skipping text feature processing.")

# -----------------------------------------------------------------------------------
# 10. Feature Selection and Separation for Tasks
# -----------------------------------------------------------------------------------
print("\n--- Feature Selection and Separation for Tasks ---")

# Dropping unnecessary columns
columns_to_drop = ['url', 'sku', 'name', 'description', 'scraped_at', 'item_id']
data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

print("\nColumns After Dropping Unnecessary Ones:")
print(data.columns.tolist())

# -----------------------------------------------------------------------------------
# 11. Ensure All Features Are Numeric Before Scaling
# -----------------------------------------------------------------------------------
print("\n--- Ensuring All Features Are Numeric ---")

# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns.tolist()
print("\nNon-numeric columns in the dataset before encoding:")
print(non_numeric_columns)

# Encode any remaining non-numeric columns
if non_numeric_columns:
    print("\nEncoding remaining non-numeric columns...")
    # Using One-Hot Encoding for simplicity
    data = pd.get_dummies(data, columns=non_numeric_columns, drop_first=False)
    print("Encoding completed.")
else:
    print("No non-numeric columns to encode.")

# Verify that there are no non-numeric columns left
non_numeric_columns_after = data.select_dtypes(include=['object']).columns.tolist()
print("\nNon-numeric columns after encoding:")
print(non_numeric_columns_after)

# -----------------------------------------------------------------------------------
# 12. Feature Selection with Recursive Feature Elimination and Cross-Validation (RFECV)
# -----------------------------------------------------------------------------------
from sklearn.feature_selection import RFECV

print("\n--- Feature Selection with RFECV ---")

# Define target variable for stockout prediction
if 'will_stockout' in data.columns:
    X_stockout = data.drop(['will_stockout'], axis=1)
    y_stockout = data['will_stockout']
else:
    logging.error("Target variable 'will_stockout' not found in the dataset.")
    raise KeyError("Target variable 'will_stockout' not found in the dataset.")

# Feature Scaling for stockout prediction
scaler_stockout = StandardScaler()
X_stockout_scaled = scaler_stockout.fit_transform(X_stockout)

# Convert scaled features back to DataFrame for feature selection
X_stockout_scaled_df = pd.DataFrame(X_stockout_scaled, columns=X_stockout.columns, index=X_stockout.index)

# Check for duplicated features and remove them
X_stockout_scaled_df = X_stockout_scaled_df.loc[:, ~X_stockout_scaled_df.columns.duplicated()]

# Feature Selection using RFECV with Logistic Regression
model_selector_stockout = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Initialize RFECV
rfecv = RFECV(estimator=model_selector_stockout, step=1, cv=cv_strategy, scoring='f1', n_jobs=-1)

# Fit RFECV on preprocessed data
rfecv.fit(X_stockout_scaled_df, y_stockout)

# Get the selected features
selected_features_stockout = X_stockout_scaled_df.columns[rfecv.support_].tolist()

print(f"\nOptimal number of features: {rfecv.n_features_}")
print(f"Selected features: {selected_features_stockout}")

# Update X_selected with selected features for stockout prediction
X_selected_stockout = X_stockout_scaled_df[selected_features_stockout]

# -----------------------------------------------------------------------------------
# 13. Train-Test Split for Stockout Prediction
# -----------------------------------------------------------------------------------
print("\n--- Train-Test Split for Stockout Prediction ---")

X_train_stockout, X_test_stockout, y_train_stockout, y_test_stockout = train_test_split(
    X_selected_stockout, y_stockout, test_size=0.2, stratify=y_stockout, random_state=RANDOM_STATE
)

print("\nClass Distribution in Updated Training Set:")
print(y_train_stockout.value_counts())

print("\nClass Distribution in Testing Set:")
print(y_test_stockout.value_counts())

# -----------------------------------------------------------------------------------
# 14. Handling Class Imbalance with SMOTE for Stockout Prediction
# -----------------------------------------------------------------------------------
print("\n--- Handling Class Imbalance with SMOTE for Stockout Prediction ---")

# Apply SMOTE to oversample the minority class in training set
smote = SMOTE(random_state=RANDOM_STATE)
X_train_stockout_res, y_train_stockout_res = smote.fit_resample(X_train_stockout, y_train_stockout)

print("\nClass Distribution After SMOTE Resampling (Updated):")
print(pd.Series(y_train_stockout_res).value_counts())

# -----------------------------------------------------------------------------------
# 15. Inventory Inaccuracy Prediction Pipeline
# -----------------------------------------------------------------------------------
print("\n--- Inventory Inaccuracy Prediction Pipeline ---")

# Define Features and Target for Inaccuracy Rate
if 'inaccuracy_rate' in data.columns:
    X_inaccuracy = data.drop(['inaccuracy_rate', 'actual_stock', 'will_stockout'], axis=1, errors='ignore')
    y_inaccuracy = data['inaccuracy_rate']

    # Feature Scaling for inaccuracy prediction
    scaler_inaccuracy = StandardScaler()
    X_inaccuracy_scaled = scaler_inaccuracy.fit_transform(X_inaccuracy)

    # Convert scaled features back to DataFrame for feature selection
    X_inaccuracy_scaled_df = pd.DataFrame(X_inaccuracy_scaled, columns=X_inaccuracy.columns, index=X_inaccuracy.index)

    # Check for duplicated features and remove them
    X_inaccuracy_scaled_df = X_inaccuracy_scaled_df.loc[:, ~X_inaccuracy_scaled_df.columns.duplicated()]

    # Feature Selection using RFECV with Random Forest Regressor
    model_selector_inaccuracy = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)

    # Initialize RFECV
    rfecv_inaccuracy = RFECV(estimator=model_selector_inaccuracy, step=1, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit RFECV on preprocessed data
    rfecv_inaccuracy.fit(X_inaccuracy_scaled_df, y_inaccuracy)

    # Get the selected features
    selected_features_inaccuracy = X_inaccuracy_scaled_df.columns[rfecv_inaccuracy.support_].tolist()

    print(f"\nOptimal number of features for Inaccuracy Prediction: {rfecv_inaccuracy.n_features_}")
    print(f"Selected features for Inaccuracy Prediction: {selected_features_inaccuracy}")

    # Update X_selected_inaccuracy with selected features
    X_selected_inaccuracy = X_inaccuracy_scaled_df[selected_features_inaccuracy]

    # -----------------------------------------------------------------------------------
    # 16. Train-Test Split for Inventory Inaccuracy Prediction
    # -----------------------------------------------------------------------------------
    print("\n--- Train-Test Split for Inventory Inaccuracy Prediction ---")

    X_train_inaccuracy, X_test_inaccuracy, y_train_inaccuracy, y_test_inaccuracy = train_test_split(
        X_selected_inaccuracy, y_inaccuracy, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\nStatistics of Training Set for Inaccuracy Prediction:")
    print(y_train_inaccuracy.describe())

    print("\nStatistics of Testing Set for Inaccuracy Prediction:")
    print(y_test_inaccuracy.describe())
else:
    print("\nWarning: 'inaccuracy_rate' column not found in the dataset. Skipping inventory inaccuracy prediction.")
    X_train_inaccuracy = X_test_inaccuracy = y_train_inaccuracy = y_test_inaccuracy = None

# -----------------------------------------------------------------------------------
# 17. Hyperparameter Tuning Functions with Optuna
# -----------------------------------------------------------------------------------
def optuna_tuning(model, param_distributions, X, y, n_trials=50, cv=5, scoring='f1', timeout=None):
    """
    Perform hyperparameter tuning using Optuna with pruning and logging.

    Parameters:
    - model: The machine learning model to tune.
    - param_distributions: Dictionary of hyperparameters to tune.
    - X: Feature matrix.
    - y: Target vector.
    - n_trials: Number of Optuna trials.
    - cv: Number of cross-validation folds.
    - scoring: Scoring metric.
    - timeout: Maximum time for the study (in seconds).

    Returns:
    - best_model: Model with the best hyperparameters.
    """
    def objective(trial):
        params = {}
        if isinstance(model, LogisticRegression):
            # Combine 'penalty' and 'solver' into a single parameter to avoid invalid combinations
            penalty_solver = trial.suggest_categorical('penalty_solver', [
                'l1_liblinear', 'l1_saga',
                'l2_lbfgs', 'l2_liblinear', 'l2_saga'
            ])
            penalty, solver = penalty_solver.split('_')
            # Suggest 'C' parameter with log scaling, ensuring C > 0
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            params = {
                'penalty': penalty,
                'solver': solver,
                'C': C,
                'max_iter': 1000
            }
        else:
            # Iterate through hyperparameters and suggest values
            for key, values in param_distributions.items():
                if isinstance(values, list):
                    params[key] = trial.suggest_categorical(key, values)
                elif isinstance(values, tuple) and len(values) == 3 and values[2] == 'int':
                    params[key] = trial.suggest_int(key, values[0], values[1])
                elif isinstance(values, tuple) and len(values) == 3 and values[2] == 'float':
                    # Ensure low > 0 when log=True
                    if 'learning_rate' in key or 'gamma' in key or 'subsample' in key or 'colsample_bytree' in key or 'reg_alpha' in key or 'reg_lambda' in key or 'bagging_temperature' in key:
                        low = max(values[0], 1e-5)  # Ensure low > 0
                        params[key] = trial.suggest_float(key, low, values[1], log=True)
                    else:
                        params[key] = trial.suggest_float(key, values[0], values[1], log=False)
                else:
                    params[key] = values
        clf = model.set_params(**params)

        # Adjust n_jobs to prevent over-utilization
        if isinstance(clf, SVC):
            n_jobs_cv = 1
        else:
            n_jobs_cv = -1  # Use all available cores for other models

        try:
            # Perform cross-validation
            score = cross_val_score(clf, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv).mean()
        except Exception as e:
            # In case of any error during fitting, return a poor score
            logging.warning(f"Trial failed with parameters {params}: {e}")
            return 0.0
        return score

    # Initialize Optuna study with TPE sampler and enable pruning
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Retrieve best parameters and score
    best_params = study.best_trial.params
    best_score = study.best_trial.value
    logging.info(f"\nBest parameters: {study.best_trial.params}")
    logging.info(f"Best {scoring}: {study.best_trial.value:.4f}")

    # Handle 'penalty_solver' if Logistic Regression
    if isinstance(model, LogisticRegression):
        penalty_solver = best_params.pop('penalty_solver')
        penalty, solver = penalty_solver.split('_')
        best_params['penalty'] = penalty
        best_params['solver'] = solver

    # Fit the model with best parameters on the entire training dataset
    try:
        best_model = model.set_params(**best_params)
        best_model.fit(X, y)
    except Exception as e:
        logging.error(f"Error fitting the model with best parameters: {e}")
        return None

    return best_model

def optuna_tuning_regression(model, param_distributions, X, y, n_trials=50, cv=5, scoring='neg_mean_squared_error', timeout=None):
    """
    Perform hyperparameter tuning using Optuna for regression models.

    Parameters:
    - model: The regression model to tune.
    - param_distributions: Dictionary of hyperparameters to tune.
    - X: Feature matrix (Training data).
    - y: Target vector (Training data).
    - n_trials: Number of Optuna trials.
    - cv: Number of cross-validation folds.
    - scoring: Scoring metric.
    - timeout: Maximum time for the study (in seconds).

    Returns:
    - best_model: Model with the best hyperparameters.
    """
    def objective(trial):
        params = {}
        # Iterate through hyperparameters and suggest values
        for key, values in param_distributions.items():
            if isinstance(values, list):
                params[key] = trial.suggest_categorical(key, values)
            elif isinstance(values, tuple) and len(values) == 3 and values[2] == 'int':
                params[key] = trial.suggest_int(key, values[0], values[1])
            elif isinstance(values, tuple) and len(values) == 3 and values[2] == 'float':
                # Ensure low > 0 when log=True
                if 'learning_rate' in key or 'gamma' in key or 'subsample' in key or 'colsample_bytree' in key or 'reg_alpha' in key or 'reg_lambda' in key or 'bagging_temperature' in key:
                    low = max(values[0], 1e-5)  # Ensure low > 0
                    params[key] = trial.suggest_float(key, low, values[1], log=True)
                else:
                    params[key] = trial.suggest_float(key, values[0], values[1], log=False)
            else:
                params[key] = values
        reg = model.set_params(**params)

        # Adjust n_jobs to prevent over-utilization
        if isinstance(reg, SVC):
            n_jobs_cv = 1
        else:
            n_jobs_cv = -1  # Use all available cores for other models

        try:
            # Perform cross-validation
            score = cross_val_score(reg, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs_cv).mean()
        except Exception as e:
            # In case of any error during fitting, return a poor score
            logging.warning(f"Trial failed with parameters {params}: {e}")
            return float('inf')  # For MSE, lower is better
        return score

    # Initialize Optuna study with TPE sampler and enable pruning
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Retrieve best parameters and score
    best_params = study.best_trial.params
    best_score = study.best_trial.value
    logging.info(f"\nBest parameters: {study.best_trial.params}")
    logging.info(f"Best {scoring}: {study.best_trial.value:.4f}")

    # Fit the model with best parameters on the entire training dataset
    try:
        best_model = model.set_params(**best_params)
        best_model.fit(X, y)
    except Exception as e:
        logging.error(f"Error fitting the model with best parameters: {e}")
        return None

    return best_model

# -----------------------------------------------------------------------------------
# 18. Implementing Stacking Classifiers for Ensemble Techniques
# -----------------------------------------------------------------------------------
print("\n--- Implementing Stacking Classifiers for Ensemble Techniques ---")

# Define base learners for stacking
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('gbc', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ('xgb', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        tree_method='gpu_hist' if GPU_AVAILABLE else 'auto',
        verbosity=0  # Suppress XGBoost warnings
    )),
    ('lgbm', LGBMClassifier(
        random_state=RANDOM_STATE,
        device_type='gpu' if GPU_AVAILABLE else 'cpu',
        verbose=-1  # Suppress LightGBM warnings
    )),
    ('catboost', CatBoostClassifier(
        verbose=0,
        random_state=RANDOM_STATE,
        task_type='GPU' if GPU_AVAILABLE else 'CPU'
    ))
]

# Define meta-learner
meta_learner = LogisticRegression(random_state=RANDOM_STATE)

# Initialize Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# -----------------------------------------------------------------------------------
# 19. Model Training and Hyperparameter Tuning for Stockout Prediction
# -----------------------------------------------------------------------------------
print("\n--- Model Training and Hyperparameter Tuning for Stockout Prediction ---")

# Initialize a dictionary to store best classification models
best_models_stockout = {}

# Define hyperparameter grids (optimized for speed and performance)
param_grids_stockout = {
    'Random Forest': {
        'n_estimators': (100, 300, 'int'),
        'max_depth': (10, 50, 'int'),
        'min_samples_split': (2, 20, 'int'),
        'min_samples_leaf': (1, 10, 'int'),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced']
    },
    'Gradient Boosting': {
        'n_estimators': (100, 300, 'int'),
        'learning_rate': (0.01, 0.3, 'float'),
        'max_depth': (3, 15, 'int'),
        'subsample': (0.5, 1.0, 'float'),
        'min_samples_split': (2, 20, 'int'),
        'min_samples_leaf': (1, 10, 'int'),
        'max_features': ['sqrt', 'log2', None]
    },
    'Support Vector Machine': {
        'C': (0.1, 10, 'float'),
        'gamma': (0.001, 1, 'float'),
        'kernel': ['linear', 'rbf']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': (3, 15, 'int'),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Multi-Layer Perceptron': {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': (0.0001, 0.01, 'float'),
        'learning_rate': ['constant', 'adaptive']
    },
    'XGBoost': {
        'n_estimators': (100, 300, 'int'),
        'learning_rate': (0.01, 0.3, 'float'),
        'max_depth': (3, 15, 'int'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'gamma': (0.001, 5, 'float'),
        'reg_alpha': (0.001, 1, 'float'),
        'reg_lambda': (0.001, 1, 'float')
    },
    'LightGBM': {
        'n_estimators': (100, 300, 'int'),
        'learning_rate': (0.01, 0.3, 'float'),
        'num_leaves': (20, 100, 'int'),
        'max_depth': (-1, 50, 'int'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'reg_alpha': (0.001, 1, 'float'),
        'reg_lambda': (0.001, 1, 'float')
    },
    'CatBoost': {
        'iterations': (100, 300, 'int'),
        'learning_rate': (0.01, 0.3, 'float'),
        'depth': (3, 10, 'int'),
        'l2_leaf_reg': (1, 10, 'int'),
        'border_count': (32, 255, 'int'),
        'bagging_temperature': (0.001, 1, 'float')
    }
}

# Define models with adjusted LightGBM and CatBoost device_type parameter
models_stockout = {
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Multi-Layer Perceptron': MLPClassifier(max_iter=1000, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        tree_method='gpu_hist' if GPU_AVAILABLE else 'auto',
        verbosity=0  # Suppress XGBoost warnings
    ),
    'LightGBM': LGBMClassifier(
        random_state=RANDOM_STATE,
        device_type='gpu' if GPU_AVAILABLE else 'cpu',
        verbose=-1  # Suppress LightGBM warnings
    ),
    'CatBoost': CatBoostClassifier(
        verbose=0,
        random_state=RANDOM_STATE,
        task_type='GPU' if GPU_AVAILABLE else 'CPU'
    )
}

# Iterate through models and perform hyperparameter tuning
for model_name, model in models_stockout.items():
    print(f"\nStarting hyperparameter tuning for {model_name}...")
    param_dist = param_grids_stockout.get(model_name, {})

    # Adjust n_trials based on the model to prevent excessive computation
    if model_name in ['CatBoost', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
        current_n_trials = 30  # Increased trials for gradient boosting models
    else:
        current_n_trials = 20  # Default number of trials for other models

    try:
        # Perform hyperparameter tuning using Optuna
        if model_name in ['Random Forest', 'Gradient Boosting', 'Support Vector Machine',
                         'K-Nearest Neighbors', 'Multi-Layer Perceptron']:
            best_model = optuna_tuning(
                model, param_dist, X_train_stockout_res, y_train_stockout_res,
                n_trials=current_n_trials, cv=5, scoring='f1', timeout=1800  # 30 minutes timeout
            )
        elif model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            # For gradient boosting models, prefer 'roc_auc' as scoring
            best_model = optuna_tuning(
                model, param_dist, X_train_stockout_res, y_train_stockout_res,
                n_trials=current_n_trials, cv=5, scoring='roc_auc', timeout=1800
            )
        else:
            best_model = optuna_tuning(
                model, param_dist, X_train_stockout_res, y_train_stockout_res,
                n_trials=current_n_trials, cv=5, scoring='f1', timeout=1800
            )

        if best_model is not None:
            best_models_stockout[model_name] = best_model
            print(f"Best {model_name} model tuned successfully.")
        else:
            print(f"Failed to tune {model_name}. Using default parameters.")
            best_models_stockout[model_name] = model.fit(X_train_stockout_res, y_train_stockout_res)
    except Exception as e:
        print(f"An error occurred during tuning of {model_name}: {e}")
        print(f"Falling back to default parameters for {model_name}.")
        best_models_stockout[model_name] = model.fit(X_train_stockout_res, y_train_stockout_res)

# -----------------------------------------------------------------------------------
# 20. Implementing Stacking Classifiers for Ensemble Techniques (Updated)
# -----------------------------------------------------------------------------------
print("\n--- Implementing Stacking Classifiers for Ensemble Techniques ---")

# Initialize Stacking Classifier with best models as base learners
stacking_estimators = []
for model_name, model in best_models_stockout.items():
    if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'Support Vector Machine', 'K-Nearest Neighbors', 'Multi-Layer Perceptron']:
        stacking_estimators.append((model_name, model))

# Define meta-learner
meta_learner = LogisticRegression(random_state=RANDOM_STATE)

# Initialize Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# Fit the stacking classifier
print("\nTraining the Stacking Classifier...")
stacking_clf.fit(X_train_stockout_res, y_train_stockout_res)
print("Stacking Classifier trained successfully.")

# Add the Stacking Classifier to the best_models_stockout dictionary
best_models_stockout['Stacking Classifier'] = stacking_clf

# -----------------------------------------------------------------------------------
# 21. Comprehensive Evaluation of All Models for Stockout Prediction
# -----------------------------------------------------------------------------------
print("\n--- Comprehensive Evaluation of All Models for Stockout Prediction ---")

# Initialize list to store all results
all_results_stockout = []

# Function to evaluate classification models
def evaluate_classification_model(name, model, X_test, y_test, threshold=0.5):
    """
    Evaluate the classification model on test data and compute various metrics.

    Parameters:
    - name: Name of the model.
    - model: Trained model.
    - X_test: Test features.
    - y_test: True labels.
    - threshold: Probability threshold for classification.

    Returns:
    - Dictionary containing evaluation metrics.
    """
    try:
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
            # Min-max scaling to convert decision scores to probabilities
            y_probs = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
        else:
            logging.warning(f"Model '{name}' does not support probability estimates.")
            y_probs = np.zeros_like(y_test)

        y_pred = (y_probs >= threshold).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # Calculate ROC AUC and PR AUC, handling edge cases
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
        else:
            roc_auc = 0.5
            pr_auc = 0.0

        # Generate classification report and confusion matrix
        class_report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return {
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'MCC': mcc,
            'Cohen\'s Kappa': kappa,
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc,
            'Threshold': threshold,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix,
            'y_probs': y_probs  # Stored for ROC Curve plotting
        }
    except Exception as e:
        logging.error(f"Error evaluating model '{name}': {e}")
        return None

# Evaluate all classification models
for model_name, model in best_models_stockout.items():
    print(f"\nEvaluating {model_name}...")
    result = evaluate_classification_model(model_name, model, X_test_stockout, y_test_stockout, threshold=0.5)
    if result:
        all_results_stockout.append(result)
        # Print Classification Report
        print(f"\n--- Classification Report for {model_name} ---\n")
        print(result['Classification Report'])

        # Plot Confusion Matrix
        cm = result['Confusion Matrix']
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        # Plot ROC Curve
        if np.any(result['y_probs']):
            fpr, tpr, thresholds = roc_curve(y_test_stockout, result['y_probs'])
            roc_auc = roc_auc_score(y_test_stockout, result['y_probs'])
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            plt.plot([0,1], [0,1], 'k--', label='Random Classifier (AUC = 0.50)')
            plt.title(f'ROC Curve for {model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Model '{model_name}' does not have probability estimates for ROC Curve plotting.")
    else:
        print(f"Skipping evaluation for {model_name} due to errors.")

# Convert results to DataFrame
results_df_stockout = pd.DataFrame(all_results_stockout)

# Identify Top 2 and Worst 2 Performers based on F1-Score
if not results_df_stockout.empty:
    top_performers_stockout = results_df_stockout.sort_values(by='F1-Score', ascending=False).head(2)
    worst_performers_stockout = results_df_stockout.sort_values(by='F1-Score').head(2)

    print("\nBenchmark of Model Performance for Stockout Prediction:")
    print(results_df_stockout[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC',
                         'Cohen\'s Kappa', 'ROC AUC', 'PR AUC', 'Threshold']])

    print("\nTop 2 Performers:")
    print(top_performers_stockout[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC',
                         'Cohen\'s Kappa', 'ROC AUC', 'PR AUC', 'Threshold']])

    print("\nWorst 2 Performers:")
    print(worst_performers_stockout[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC',
                         'Cohen\'s Kappa', 'ROC AUC', 'PR AUC', 'Threshold']])
else:
    print("\nNo models were evaluated successfully for Stockout Prediction.")

# -----------------------------------------------------------------------------------
# 22. Comprehensive Evaluation of All Models for Inventory Inaccuracy Prediction
# -----------------------------------------------------------------------------------
if X_train_inaccuracy is not None:
    print("\n--- Comprehensive Evaluation of All Models for Inventory Inaccuracy Prediction ---")

    # Initialize a dictionary to store best regression models
    best_models_inaccuracy = {}

    # Define hyperparameter grids for regression models (similar to classification)
    param_grids_inaccuracy = {
        'Random Forest': {
            'n_estimators': (100, 300, 'int'),
            'max_depth': (10, 50, 'int'),
            'min_samples_split': (2, 20, 'int'),
            'min_samples_leaf': (1, 10, 'int'),
            'max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'n_estimators': (100, 300, 'int'),
            'learning_rate': (0.01, 0.3, 'float'),
            'max_depth': (3, 15, 'int'),
            'subsample': (0.5, 1.0, 'float'),
            'min_samples_split': (2, 20, 'int'),
            'min_samples_leaf': (1, 10, 'int'),
            'max_features': ['sqrt', 'log2', None]
        },
        'Support Vector Regressor': {
            'C': (0.1, 10, 'float'),
            'gamma': (0.001, 1, 'float'),
            'kernel': ['linear', 'rbf']
        },
        'K-Nearest Neighbors Regressor': {
            'n_neighbors': (3, 15, 'int'),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'Multi-Layer Perceptron Regressor': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': (0.0001, 0.01, 'float'),
            'learning_rate': ['constant', 'adaptive']
        },
        'XGBoost Regressor': {
            'n_estimators': (100, 300, 'int'),
            'learning_rate': (0.01, 0.3, 'float'),
            'max_depth': (3, 15, 'int'),
            'subsample': (0.5, 1.0, 'float'),
            'colsample_bytree': (0.5, 1.0, 'float'),
            'gamma': (0.001, 5, 'float'),
            'reg_alpha': (0.001, 1, 'float'),
            'reg_lambda': (0.001, 1, 'float')
        },
        'LightGBM Regressor': {
            'n_estimators': (100, 300, 'int'),
            'learning_rate': (0.01, 0.3, 'float'),
            'num_leaves': (20, 100, 'int'),
            'max_depth': (-1, 50, 'int'),
            'subsample': (0.5, 1.0, 'float'),
            'colsample_bytree': (0.5, 1.0, 'float'),
            'reg_alpha': (0.001, 1, 'float'),
            'reg_lambda': (0.001, 1, 'float')
        },
        'CatBoost Regressor': {
            'iterations': (100, 300, 'int'),
            'learning_rate': (0.01, 0.3, 'float'),
            'depth': (3, 10, 'int'),
            'l2_leaf_reg': (1, 10, 'int'),
            'border_count': (32, 255, 'int'),
            'bagging_temperature': (0.001, 1, 'float')
        }
    }

    # Define regression models with corrected definitions
    models_inaccuracy = {
        'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
        'Support Vector Regressor': SVR(kernel='rbf'),  # Removed 'random_state' as SVR doesn't use it
        'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
        'Multi-Layer Perceptron Regressor': MLPRegressor(max_iter=1000, random_state=RANDOM_STATE),
        'XGBoost Regressor': XGBRegressor(
            random_state=RANDOM_STATE,
            objective='reg:squarederror',
            tree_method='gpu_hist' if GPU_AVAILABLE else 'auto',
            verbosity=0  # Suppress XGBoost warnings
        ),
        'LightGBM Regressor': LGBMRegressor(
            random_state=RANDOM_STATE,
            device_type='gpu' if GPU_AVAILABLE else 'cpu',
            verbose=-1  # Suppress LightGBM warnings
        ),
        'CatBoost Regressor': CatBoostRegressor(
            verbose=0,
            random_state=RANDOM_STATE,
            task_type='GPU' if GPU_AVAILABLE else 'CPU'
        )
    }

    # Iterate through regression models and perform hyperparameter tuning
    for model_name, model in models_inaccuracy.items():
        print(f"\nStarting hyperparameter tuning for {model_name}...")
        param_dist = param_grids_inaccuracy.get(model_name, {})

        # Adjust n_trials based on the model to prevent excessive computation
        if model_name in ['CatBoost Regressor', 'XGBoost Regressor', 'LightGBM Regressor', 'Gradient Boosting']:
            current_n_trials = 30  # Increased trials for gradient boosting models
        else:
            current_n_trials = 20  # Default number of trials for other models

        try:
            # Perform hyperparameter tuning using Optuna
            if model_name in ['Random Forest', 'Gradient Boosting', 'Support Vector Regressor',
                             'K-Nearest Neighbors Regressor', 'Multi-Layer Perceptron Regressor']:
                best_model = optuna_tuning_regression(
                    model, param_dist, X_train_inaccuracy, y_train_inaccuracy,
                    n_trials=current_n_trials, cv=5, scoring='neg_mean_squared_error', timeout=1800  # 30 minutes timeout
                )
            elif model_name in ['XGBoost Regressor', 'LightGBM Regressor', 'CatBoost Regressor']:
                # For regression gradient boosting models, prefer 'neg_mean_squared_error' as scoring
                best_model = optuna_tuning_regression(
                    model, param_dist, X_train_inaccuracy, y_train_inaccuracy,
                    n_trials=current_n_trials, cv=5, scoring='neg_mean_squared_error', timeout=1800
                )
            else:
                best_model = optuna_tuning_regression(
                    model, param_dist, X_train_inaccuracy, y_train_inaccuracy,
                    n_trials=current_n_trials, cv=5, scoring='neg_mean_squared_error', timeout=1800
                )

            if best_model is not None:
                best_models_inaccuracy[model_name] = best_model
                print(f"Best {model_name} model tuned successfully.")
            else:
                print(f"Failed to tune {model_name}. Using default parameters.")
                best_models_inaccuracy[model_name] = model.fit(X_train_inaccuracy, y_train_inaccuracy)
        except Exception as e:
            print(f"An error occurred during tuning of {model_name}: {e}")
            print(f"Falling back to default parameters for {model_name}.")
            best_models_inaccuracy[model_name] = model.fit(X_train_inaccuracy, y_train_inaccuracy)

    # -----------------------------------------------------------------------------------
    # 22. Comprehensive Evaluation of All Models for Inventory Inaccuracy Prediction
    # -----------------------------------------------------------------------------------
    if X_train_inaccuracy is not None:
        print("\n--- Comprehensive Evaluation of All Models for Inventory Inaccuracy Prediction ---")

        # Initialize list to store all regression results
        all_results_inaccuracy = []

        # Function to evaluate regression models
        def evaluate_regression_model(name, model, X_test, y_test):
            """
            Evaluate the regression model on test data and compute various metrics.

            Parameters:
            - name: Name of the model.
            - model: Trained regression model.
            - X_test: Test features.
            - y_test: True target values.

            Returns:
            - Dictionary containing evaluation metrics.
            """
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {
                'Model': name,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2
            }

        # Evaluate all regression models
        for model_name, model in best_models_inaccuracy.items():
            print(f"\nEvaluating {model_name}...")
            result = evaluate_regression_model(model_name, model, X_test_inaccuracy, y_test_inaccuracy)
            if result:
                all_results_inaccuracy.append(result)
                # Print regression metrics
                print(f"\n--- Regression Metrics for {model_name} ---")
                print(f"MSE: {result['MSE']:.4f}")
                print(f"RMSE: {result['RMSE']:.4f}")
                print(f"MAE: {result['MAE']:.4f}")
                print(f"R² Score: {result['R² Score']:.4f}")

                # Plot Bar Chart for Regression Metrics
                metrics = ['MSE', 'RMSE', 'MAE', 'R² Score']
                scores = [result['MSE'], result['RMSE'], result['MAE'], result['R² Score']]

                plt.figure(figsize=(8,6))
                sns.barplot(x=metrics, y=scores, palette='viridis')
                plt.title(f'Regression Metrics for {model_name}')
                plt.ylabel('Score')
                plt.xlabel('Metrics')
                plt.tight_layout()
                plt.show()
            else:
                print(f"Skipping evaluation for {model_name} due to errors.")

        # Convert results to DataFrame
        results_df_inaccuracy = pd.DataFrame(all_results_inaccuracy)

        # Identify Top 2 and Worst 2 Performers based on RMSE
        if not results_df_inaccuracy.empty:
            top_performers_inaccuracy = results_df_inaccuracy.sort_values(by='RMSE').head(2)
            worst_performers_inaccuracy = results_df_inaccuracy.sort_values(by='RMSE', ascending=False).head(2)

            print("\nBenchmark of Model Performance for Inventory Inaccuracy Prediction:")
            print(results_df_inaccuracy[['Model', 'MSE', 'RMSE', 'MAE', 'R² Score']])

            print("\nTop 2 Performers:")
            print(top_performers_inaccuracy[['Model', 'MSE', 'RMSE', 'MAE', 'R² Score']])

            print("\nWorst 2 Performers:")
            print(worst_performers_inaccuracy[['Model', 'MSE', 'RMSE', 'MAE', 'R² Score']])
        else:
            print("\nNo regression models were evaluated successfully for Inventory Inaccuracy Prediction.")
    else:
        results_df_inaccuracy = pd.DataFrame()
        print("\nNo models were tuned for Inventory Inaccuracy Prediction due to missing 'inaccuracy_rate' column.")

# -----------------------------------------------------------------------------------
# 23. Visualization of Benchmark Results for Stockout Prediction
# -----------------------------------------------------------------------------------
if not results_df_stockout.empty:
    # Plot Benchmark Visualization: F1, Recall, Precision, Accuracy, ROC AUC
    benchmark_metrics = ['F1-Score', 'Recall', 'Precision', 'Accuracy', 'ROC AUC']
    benchmark_df = results_df_stockout[['Model'] + benchmark_metrics].set_index('Model')

    plt.figure(figsize=(14, 7))
    benchmark_df.plot(kind='bar', figsize=(14, 7))
    plt.title('Benchmark of Models for Stockout Prediction')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Feature Importance and SHAP Analysis for Top 2 Performers
    for index, row in top_performers_stockout.iterrows():
        best_model_name_stockout = row['Model']
        best_model_final_stockout = best_models_stockout.get(best_model_name_stockout)
        print(f"\n--- Feature Importance and SHAP Analysis for {best_model_name_stockout} ---")
        if best_model_final_stockout and hasattr(best_model_final_stockout, 'feature_importances_'):
            try:
                # Feature Importance
                importances = best_model_final_stockout.feature_importances_
                feature_importance = pd.Series(importances, index=X_selected_stockout.columns)
                feature_importance = feature_importance.sort_values(ascending=False).head(20)

                plt.figure(figsize=(12,8))
                sns.barplot(x=feature_importance.values, y=feature_importance.index)
                plt.title(f'Top 20 Feature Importances for {best_model_name_stockout}')
                plt.xlabel('Importance Score')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.show()

                # SHAP Analysis
                if isinstance(best_model_final_stockout, (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
                    explainer = shap.TreeExplainer(best_model_final_stockout)
                    shap_values = explainer.shap_values(X_test_stockout)

                    # SHAP Summary Plot (Bar)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test_stockout, plot_type="bar", show=False)
                    plt.title(f'SHAP Summary Plot for {best_model_name_stockout}')
                    plt.tight_layout()
                    plt.show()

                    # SHAP Summary Plot (Detailed)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test_stockout, show=False)
                    plt.title(f'SHAP Summary Plot (Detailed) for {best_model_name_stockout}')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"SHAP analysis not supported for model type: {type(best_model_final_stockout)}")
            except Exception as e:
                print(f"\nAn error occurred during SHAP analysis: {e}")
        elif best_model_final_stockout and hasattr(best_model_final_stockout, 'coef_'):
            # For models like Logistic Regression
            try:
                coef = best_model_final_stockout.coef_[0]
                feature_importance = pd.Series(coef, index=X_selected_stockout.columns).abs().sort_values(ascending=False).head(20)
                plt.figure(figsize=(12,8))
                sns.barplot(x=feature_importance.values, y=feature_importance.index)
                plt.title(f'Top 20 Feature Coefficients for {best_model_name_stockout}')
                plt.xlabel('Coefficient Magnitude')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"\nAn error occurred while plotting coefficients: {e}")
        else:
            print(f"\nSHAP analysis not supported for model type: {type(best_model_final_stockout)}")

    # Improved Correlation Heatmap
    print("\n--- Correlation Heatmap ---")
    plt.figure(figsize=(16, 12))
    corr = data.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of All Features')
    plt.tight_layout()
    plt.show()

    # Visualization for Top 2 Best Models Selected for Stockout Prediction
    print("\n--- Visualization for Top 2 Best Models Selected for Stockout Prediction ---")
    for model_name in top_performers_stockout['Model']:
        model = best_models_stockout.get(model_name)
        if model and hasattr(model, 'feature_importances_'):
            try:
                # SHAP Dependence Plot for Top Features
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_stockout)

                top_features = X_selected_stockout.columns[np.argsort(model.feature_importances_)[::-1][:5]].tolist()

                for feature in top_features:
                    plt.figure(figsize=(8,6))
                    shap.dependence_plot(feature, shap_values, X_test_stockout, show=False)
                    plt.title(f'SHAP Dependence Plot for {feature} in {model_name}')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"\nAn error occurred during SHAP dependence plot: {e}")
        elif model and hasattr(model, 'coef_'):
            # For Logistic Regression
            try:
                coef = model.coef_[0]
                feature_importance = pd.Series(coef, index=X_selected_stockout.columns).abs().sort_values(ascending=False).head(5)
                for feature in feature_importance.index:
                    plt.figure(figsize=(8,6))
                    shap.dependence_plot(feature, shap_values, X_test_stockout, show=False)
                    plt.title(f'SHAP Dependence Plot for {feature} in {model_name}')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"\nAn error occurred during coefficient dependence plot: {e}")
        else:
            print(f"\nNo suitable model available for SHAP dependence plot for {model_name}.")

    # RFID Impact on Stockout Prediction
    print("\n--- RFID Impact on Stockout Prediction ---")
    if 'RFID_Tag' in data.columns and 'will_stockout' in data.columns:
        plt.figure(figsize=(8,6))
        sns.countplot(x='RFID_Tag', hue='will_stockout', data=data)
        plt.title('Impact of RFID on Stockout Occurrence')
        plt.xlabel('RFID Tag (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.legend(title='Will Stockout', labels=['No', 'Yes'])
        plt.tight_layout()
        plt.show()
    else:
        print("Columns 'RFID_Tag' and/or 'will_stockout' not found. Skipping RFID impact plot.")
else:
    print("\nNo results to plot for Stockout Prediction.")

# -----------------------------------------------------------------------------------
# 25. Visualization of Benchmark Results for Inventory Inaccuracy Prediction
# -----------------------------------------------------------------------------------
if not results_df_inaccuracy.empty:
    # Plot Benchmark Visualization: RMSE, MAE, R² Score
    benchmark_metrics_inaccuracy = ['RMSE', 'MAE', 'R² Score']
    benchmark_df_inaccuracy = results_df_inaccuracy[['Model'] + benchmark_metrics_inaccuracy].set_index('Model')

    plt.figure(figsize=(14, 7))
    benchmark_df_inaccuracy.plot(kind='bar', figsize=(14, 7))
    plt.title('Benchmark of Models for Inventory Inaccuracy Prediction')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Feature Importance and SHAP Analysis for Top 2 Performers
    for index, row in top_performers_inaccuracy.iterrows():
        best_model_inaccuracy_name = row['Model']
        best_model_inaccuracy = best_models_inaccuracy.get(best_model_inaccuracy_name)
        print(f"\n--- Feature Importance and SHAP Analysis for {best_model_inaccuracy_name} ---")
        if best_model_inaccuracy and hasattr(best_model_inaccuracy, 'feature_importances_'):
            try:
                # Feature Importance
                importances = best_model_inaccuracy.feature_importances_
                feature_importance = pd.Series(importances, index=X_selected_inaccuracy.columns)
                feature_importance = feature_importance.sort_values(ascending=False).head(20)

                plt.figure(figsize=(12,8))
                sns.barplot(x=feature_importance.values, y=feature_importance.index)
                plt.title(f'Top 20 Feature Importances for {best_model_inaccuracy_name}')
                plt.xlabel('Importance Score')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.show()

                # SHAP Analysis
                if isinstance(best_model_inaccuracy, (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, LGBMRegressor, CatBoostRegressor)):
                    explainer_inaccuracy = shap.TreeExplainer(best_model_inaccuracy)
                    shap_values_inaccuracy = explainer_inaccuracy.shap_values(X_test_inaccuracy)

                    # SHAP Summary Plot (Bar)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_inaccuracy, X_test_inaccuracy, plot_type="bar", show=False)
                    plt.title(f'SHAP Summary Plot for {best_model_inaccuracy_name} - Inventory Inaccuracy')
                    plt.tight_layout()
                    plt.show()

                    # SHAP Summary Plot (Detailed)
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_inaccuracy, X_test_inaccuracy, show=False)
                    plt.title(f'SHAP Summary Plot (Detailed) for {best_model_inaccuracy_name} - Inventory Inaccuracy')
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"SHAP analysis not supported for model type: {type(best_model_inaccuracy)}")
            except Exception as e:
                print(f"\nAn error occurred during SHAP analysis for Inventory Inaccuracy: {e}")
        else:
            print(f"\nSHAP analysis not supported for model type: {type(best_model_inaccuracy)}")

    # Visualization for Top 2 Best Models Selected for Inventory Inaccuracy Prediction
    print("\n--- Visualization for Top 2 Best Models Selected for Inventory Inaccuracy Prediction ---")
    for model_name in top_performers_inaccuracy['Model']:
        model = best_models_inaccuracy.get(model_name)
        if model and hasattr(model, 'feature_importances_'):
            try:
                # SHAP Dependence Plot for Top Features
                explainer_inaccuracy = shap.TreeExplainer(model)
                shap_values_inaccuracy = explainer_inaccuracy.shap_values(X_test_inaccuracy)

                importances_inaccuracy_best = model.feature_importances_
                feature_importance_inaccuracy = pd.Series(importances_inaccuracy_best, index=X_selected_inaccuracy.columns)
                feature_importance_inaccuracy = feature_importance_inaccuracy.sort_values(ascending=False).head(5)
                top_features_inaccuracy = feature_importance_inaccuracy.index.tolist()

                for feature in top_features_inaccuracy:
                    plt.figure(figsize=(8,6))
                    shap.dependence_plot(feature, shap_values_inaccuracy, X_test_inaccuracy, show=False)
                    plt.title(f'SHAP Dependence Plot for {feature} in {model_name}')
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                print(f"\nAn error occurred during SHAP dependence plot: {e}")
        else:
            print(f"\nNo suitable model available for SHAP dependence plot for {model_name}.")

    # RFID Impact on Inventory Inaccuracy
    print("\n--- RFID Impact on Inventory Inaccuracy ---")
    if 'RFID_Tag' in data.columns and 'inaccuracy_rate' in data.columns:
        plt.figure(figsize=(8,6))
        sns.boxplot(x='RFID_Tag', y='inaccuracy_rate', data=data)
        plt.title('Impact of RFID on Inventory Inaccuracy Rate')
        plt.xlabel('RFID Tag (0 = No, 1 = Yes)')
        plt.ylabel('Inaccuracy Rate')
        plt.tight_layout()
        plt.show()
    else:
        print("Columns 'RFID_Tag' and/or 'inaccuracy_rate' not found. Skipping RFID impact plot.")
else:
    print("\nNo results to plot for Inventory Inaccuracy Prediction.")

# -----------------------------------------------------------------------------------
# 26. Saving the Best Models
# -----------------------------------------------------------------------------------
print("\n--- Saving the Best Models ---")

# Selecting the best classification models (Top 2)
if not results_df_stockout.empty:
    best_model_names_stockout = top_performers_stockout['Model'].tolist()
    best_models_final_stockout = {name: best_models_stockout[name] for name in best_model_names_stockout}
    print(f"\nTop 2 Best Models Selected for Stockout Prediction: {best_model_names_stockout}")
else:
    print("\nNo models were evaluated successfully for Stockout Prediction.")
    best_models_final_stockout = {}

# Selecting the best regression models (Top 2)
if not results_df_inaccuracy.empty:
    best_model_names_inaccuracy = top_performers_inaccuracy['Model'].tolist()
    best_models_final_inaccuracy = {name: best_models_inaccuracy[name] for name in best_model_names_inaccuracy}
    print(f"\nTop 2 Best Models Selected for Inventory Inaccuracy Prediction: {best_model_names_inaccuracy}")
else:
    print("\nNo regression models were evaluated successfully for Inventory Inaccuracy Prediction.")
    best_models_final_inaccuracy = {}

# Save classification models
if best_models_final_stockout:
    for model_name, model in best_models_final_stockout.items():
        try:
            joblib.dump(model, f'best_stockout_model_{model_name.replace(" ", "_")}.joblib')
            print(f"Best model '{model_name}' saved successfully as 'best_stockout_model_{model_name.replace(' ', '_')}.joblib'.")
        except Exception as e:
            print(f"An error occurred while saving the stockout model '{model_name}': {e}")
else:
    print("No models available to save for Stockout Prediction.")

# Save regression models
if best_models_final_inaccuracy:
    for model_name, model in best_models_final_inaccuracy.items():
        try:
            joblib.dump(model, f'best_inaccuracy_model_{model_name.replace(" ", "_")}.joblib')
            print(f"Best model '{model_name}' saved successfully as 'best_inaccuracy_model_{model_name.replace(' ', '_')}.joblib'.")
        except Exception as e:
            print(f"An error occurred while saving the inaccuracy model '{model_name}': {e}")
else:
    print("No models available to save for Inventory Inaccuracy Prediction.")

# -----------------------------------------------------------------------------------
# 27. Additional Feature Distribution Plots
# -----------------------------------------------------------------------------------
print("\n--- Feature Distribution Plots ---")
# Plot distribution of 'price' before and after log transformation
if 'price' in data.columns and 'price_log' in data.columns:
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(data['price'], kde=True, bins=30, color='skyblue')
    plt.title('Price Distribution')

    plt.subplot(1,2,2)
    sns.histplot(data['price_log'], kde=True, bins=30, color='salmon')
    plt.title('Log-Transformed Price Distribution')
    plt.tight_layout()
    plt.show()
else:
    print("\nColumns 'price' and/or 'price_log' not found. Skipping their distribution plots.")

# Plot distribution of 'stock_level'
if 'stock_level' in data.columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data['stock_level'], kde=True, bins=30, color='green')
    plt.title('Stock Level Distribution')
    plt.xlabel('Stock Level')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
else:
    print("\nColumn 'stock_level' not found. Skipping its distribution plot.")

# Plot distribution of 'inaccuracy_rate' if available
if 'inaccuracy_rate' in data.columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data['inaccuracy_rate'], kde=True, bins=30, color='purple')
    plt.title('Inventory Inaccuracy Rate Distribution')
    plt.xlabel('Inaccuracy Rate')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
else:
    print("\n'inaccuracy_rate' column not found. Skipping its distribution plot.")

# -----------------------------------------------------------------------------------
# End of Pipeline
# -----------------------------------------------------------------------------------
