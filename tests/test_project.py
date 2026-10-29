import os
from pathlib import Path
import joblib
import pandas as pd
import pytest

from train import load_and_validate_data, build_pipeline, REQUIRED_COLUMNS

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

def test_dataset_exists_and_valid():
    """Test that heart.csv exists and loads successfully with valid schema."""
    assert DATA_PATH.exists(), "Dataset heart.csv is missing."
    df = load_and_validate_data(DATA_PATH)
    assert not df.empty, "Dataset is empty."
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column {col} missing from dataset."
    assert set(df['target'].unique()).issubset({0, 1}), "Target column contains invalid values."

def test_model_training_and_serialization():
    """Test that training pipeline runs and serializes model.pkl successfully."""
    from train import train_and_evaluate
    metrics = train_and_evaluate(DATA_PATH, MODEL_PATH)
    assert MODEL_PATH.exists(), "model.pkl was not created."
    assert 'accuracy' in metrics, "Metrics missing accuracy."
    assert 0.0 <= metrics['accuracy'] <= 1.0, "Accuracy out of bounds."

def test_model_prediction():
    """Test that the loaded model can predict on sample feature vector."""
    assert MODEL_PATH.exists(), "model.pkl missing for prediction test."
    model = joblib.load(MODEL_PATH)

    # Sample row from heart.csv (excluding target)
    sample_features = pd.DataFrame([[
        63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1
    ]], columns=REQUIRED_COLUMNS[:-1])

    prediction = model.predict(sample_features)
    assert len(prediction) == 1, "Prediction output length invalid."
    assert prediction[0] in [0, 1], "Prediction result is not binary (0 or 1)."

def test_invalid_dataset_column(tmp_path):
    """Test that invalid dataset columns raise ValueError."""
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({'invalid_col': [1, 2, 3]}).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError):
        load_and_validate_data(invalid_csv)
