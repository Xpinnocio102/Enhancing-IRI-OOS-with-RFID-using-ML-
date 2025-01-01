# Enhancing-IRI-OOS-with-RFID-using-ML-
Enhancing Inventory Accuracy and Reducing Stockouts through RFID Item-Level Tracking Data and Machine Learning 
Table of Contents

Project Overview

Installation

Usage

Pipeline Steps

Importing Libraries
Reproducibility Setup
Hardware Utilization Check
Data Loading
Exploratory Data Analysis (EDA)
Categorical Variable Encoding
Missing Values Handling
Feature Engineering
Text Feature Extraction
Feature Selection and Task Separation
Ensuring Numeric Features
Recursive Feature Elimination with Cross-Validation (RFECV)
Train-Test Split
Addressing Class Imbalance
Inventory Inaccuracy Prediction Pipeline
Hyperparameter Tuning with Optuna
Ensemble Techniques with Stacking Classifiers
Model Evaluation
Model Saving
Additional Visualizations
Results Conclusion

Project Overview

Efficient inventory management is critical for retail operations to minimize losses due to stockouts and inaccuracies. Leveraging RFID technology, this project aims to develop predictive models that:

Predict Stockouts: Identify products at risk of being out of stock. Estimate Inventory Inaccuracy: Determine discrepancies between recorded and actual stock levels.

By integrating advanced machine learning techniques, the pipeline ensures high prediction accuracy, enabling proactive inventory management and strategic decision-making.

Installation To replicate this project, the following Python libraries are installed in your environment:

Data Manipulation: pandas, numpy Machine Learning: scikit-learn, xgboost, lightgbm, catboost Visualization: matplotlib, seaborn, shap Optimization: optuna Utilities: joblib, imbalanced-learn, featuretools You can install these libraries using pip:

pip install pandas numpy scikit-learn matplotlib seaborn optuna xgboost lightgbm catboost shap joblib imbalanced-learn featuretools Note: If you are using a Jupyter Notebook, prepend the command with ! to execute it within a cell.

Usage

Prepare the Dataset:

Ensure the dataset file ZARA_RFID_File_Final.csv is accessible in your working directory or specify the correct path. Execute the Pipeline:

Run the Python script or Jupyter Notebook sequentially to perform data preprocessing, model training, and evaluation. Review Outputs:

The pipeline will generate evaluation metrics, visualizations, and save the best-performing models for deployment or further analysis. Pipeline Steps

Importing Libraries All necessary libraries for data processing, modeling, evaluation, and visualization are imported to facilitate the pipeline's operations.

Reproducibility Setup A fixed random state is set to ensure that the results are consistent across different runs, enhancing the reliability of the experiments.

Hardware Utilization Check The pipeline checks for GPU availability to leverage accelerated computing for models like XGBoost and CatBoost, which can significantly reduce training time.

Data Loading The dataset ZARA_RFID_File_Final.csv is loaded into a Pandas DataFrame. Error handling ensures that issues like missing files or incorrect formats are appropriately managed.

Exploratory Data Analysis (EDA) Initial data exploration includes:

Displaying the first few records. Encoding the target variable will_stockout into a binary format. Parsing and handling datetime features, specifically the scraped_at column.

Categorical Variable Encoding Categorical variables are transformed into numerical formats to be compatible with machine learning algorithms:

Binary Encoding: Variables like is_discounted and is_returnable are mapped to binary values. One-Hot Encoding: Features such as gender, location_type, subcategory, and category are one-hot encoded to create binary indicator variables. RFID Tag Encoding: The RFID_Tag feature is also binary encoded.

Missing Values Handling Missing numerical values in columns like price, stock_level, and actual_stock are imputed using the mean strategy to maintain data integrity.

Feature Engineering New features are created to enhance model performance:

Log Transformation: Applied to the price feature to handle skewness. Ratios and Interactions: Features like stock_price_ratio and price_stock_interaction are derived. Temporal Features: Extracted from the scraped_at datetime column, including hour of day, day of week, and weekend indicators. Polynomial Features: Squared terms for price and stock_level are added to capture non-linear relationships.

Text Feature Extraction If a description column exists, textual data is transformed using TF-IDF vectorization followed by dimensionality reduction with Truncated SVD to incorporate text-based features into the model.

Feature Selection and Task Separation Unnecessary columns are dropped, and the dataset is prepared for two primary tasks:

Stockout Prediction: Targeting the will_stockout variable. Inventory Inaccuracy Prediction: Targeting the inaccuracy_rate variable.

Ensuring Numeric Features All remaining non-numeric features are identified and encoded using one-hot encoding to ensure compatibility with machine learning models.

** Recursive Feature Elimination with Cross-Validation (RFECV)** Feature selection is performed using RFECV to identify the most significant features for each prediction task, enhancing model efficiency and performance.

Train-Test Split Data is split into training and testing sets, maintaining class distribution for stockout prediction using stratified sampling to ensure balanced evaluation.

Addressing Class Imbalance SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to balance classes, mitigating biases in model training due to imbalanced datasets.

Inventory Inaccuracy Prediction Pipeline A parallel pipeline is established for predicting inventory inaccuracy, involving feature scaling and selection tailored to regression tasks.

Hyperparameter Tuning with Optuna Optuna is utilized for efficient hyperparameter optimization of both classification and regression models, enhancing model performance through intelligent search strategies.

Ensemble Techniques with Stacking Classifiers Stacking classifiers are implemented using multiple base learners (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost) with Logistic Regression as the meta-learner to leverage the strengths of various models.

Model Evaluation Each trained model undergoes comprehensive evaluation using metrics such as Accuracy, Precision, Recall, F1-Score, ROC AUC for classification, and MSE, RMSE, MAE, R² Score for regression. Visualizations like confusion matrices and ROC curves aid in performance assessment.

Model Saving The best-performing models are serialized and saved using joblib for future deployment or analysis, ensuring reproducibility and ease of access.

Additional Visualizations Further visualizations include:

Feature Importance: Identifying key drivers in model predictions. SHAP Analysis: Understanding feature contributions at both global and local levels. Correlation Heatmaps: Visualizing relationships between features. Distribution Plots: Assessing the distribution of critical features. Results Upon execution, the pipeline provides:

Evaluation Metrics: Detailed performance metrics for each model, facilitating informed comparisons.

Visual Insights: Graphical representations of model performance, feature importance, and data distributions.

Optimized Models: Saved models with the best configurations ready for deployment to predict stockouts and inventory inaccuracies effectively.

Conclusion

This machine learning pipeline offers a robust framework for predicting inventory discrepancies and stockouts using RFID-tagged data. By integrating advanced preprocessing, feature engineering, ensemble methods, and hyperparameter optimization, the pipeline achieves high accuracy and provides actionable insights for inventory management. The comprehensive evaluation and visualization components ensure that the models are both effective and interpretable, supporting strategic decision-making in retail operations.
