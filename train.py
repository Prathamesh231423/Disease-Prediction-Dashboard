import os
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REQUIRED_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

def load_and_validate_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the heart disease dataset."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Remove duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Convert numeric types & handle missing values safely
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill or drop missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)

    # Ensure target labels are binary (0 or 1)
    if not set(df['target'].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain binary values (0 or 1).")

    return df

def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create a machine learning pipeline with scaling and Random Forest Classifier."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=5))
    ])

def train_and_evaluate(data_path: str | Path, model_output_path: str | Path) -> Dict[str, Any]:
    """Train the model pipeline, evaluate it, and save the fitted pipeline using joblib."""
    df = load_and_validate_data(data_path)

    X = df.drop(columns=['target'])
    y = df['target']

    # Stratified split to prevent data leakage and preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        'test_samples': len(y_test),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    # Save fitted model pipeline with joblib
    joblib.dump(pipeline, model_output_path)
    return metrics

if __name__ == '__main__':
    base_dir = Path(__file__).parent.resolve()
    data_file = base_dir / 'data' / 'heart.csv'
    model_file = base_dir / 'model.pkl'

    print("Training heart disease prediction model pipeline...")
    results = train_and_evaluate(data_file, model_file)
    print("\n--- Training Results ---")
    print(f"Test Samples: {results['test_samples']}")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Precision: {results['precision'] * 100:.2f}%")
    print(f"Recall: {results['recall'] * 100:.2f}%")
    print(f"F1 Score: {results['f1_score'] * 100:.2f}%")
    print("Confusion Matrix:")
    print(np.array(results['confusion_matrix']))
    print(f"\nSaved model successfully to: {model_file}")
