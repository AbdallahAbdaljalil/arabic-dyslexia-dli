"""
svr_model.py
------------
SVR regression model for predicting DLI scores from Arabic text features.
Uses LLM scores as silver labels for training.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import joblib


# ==============================================================================
# FEATURES
# ==============================================================================

FEATURE_COLS = [
    'DLP',
    'Dot_Load_Ratio',
    'OVL',
    'Similar_Shape_Density',
    'PSC_Density',
    'Lexical_Difficulty',
    'Homograph_Risk',
    'Avg_Morphological_Complexity'
]


# ==============================================================================
# TRAIN
# ==============================================================================

def train_svr(df, feature_cols=FEATURE_COLS, target_col='LLM_Score',
              test_size=0.2, random_state=42):
    """
    Train SVR on features predicting LLM silver labels.

    Args:
        df: DataFrame with features and LLM scores
        feature_cols: list of feature column names
        target_col: column to predict
        test_size: proportion for test split
        random_state: for reproducibility

    Returns:
        model, scaler, results dict
    """
    df = df.dropna(subset=feature_cols + [target_col])

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train SVR
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)

    # evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_r, pearson_p = stats.pearsonr(y_test, y_pred)

    # cross validation
    pipeline = Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

    results = {
        'mae': mae,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'cv_mae_mean': -cv_scores.mean(),
        'cv_mae_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }

    print(f"\n=== SVR Results ===")
    print(f"Train/Test: {len(X_train)}/{len(X_test)}")
    print(f"MAE:        {mae:.4f}")
    print(f"R²:         {r2:.4f}")
    print(f"Pearson r:  {pearson_r:.4f}  (p={pearson_p:.4f})")
    print(f"CV MAE:     {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, scaler, results


# ==============================================================================
# PREDICT
# ==============================================================================

def predict_svr(model, scaler, df, feature_cols=FEATURE_COLS):
    """
    Apply trained SVR to a DataFrame to get predicted scores.

    Args:
        model: trained SVR model
        scaler: fitted StandardScaler
        df: DataFrame with feature columns
        feature_cols: list of feature column names

    Returns:
        array of predicted scores
    """
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# ==============================================================================
# SAVE / LOAD
# ==============================================================================

def save_model(model, scaler, path='../results/models/svr_model.joblib'):
    """Save trained model and scaler."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"Model saved to {path}")


def load_model(path='../results/models/svr_model.joblib'):
    """Load trained model and scaler."""
    data = joblib.load(path)
    return data['model'], data['scaler']