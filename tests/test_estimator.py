
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import os
from income_estimator.estimator import IncomeEstimator
from income_estimator.exceptions import InputValidationError, EstimatorError

# Paths to your model and encoder
MODEL_PATH = os.path.join("models", "income_estimator_rf.pkl")
ENCODER_PATH = os.path.join("models", "ord_encoder.pkl")


@pytest.fixture(scope="module")
def estimator():
    """Fixture to initialize IncomeEstimator once per test module."""
    return IncomeEstimator(model_path=MODEL_PATH, encoder_path=ENCODER_PATH, threshold=0.8)


def test_model_load(estimator):
    """Test that the estimator has loaded model and encoder correctly."""
    assert estimator.model is not None
    assert estimator.encoder is not None
    assert hasattr(estimator.model, "predict_proba")


def test_valid_input_prediction(estimator):
    """Test prediction on a valid input dictionary."""
    valid_input = {
        "age": 39,
        "workclass": "State-gov",
        "education": "Bachelors",
        "years_of_education": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "ethinicity": "White",
        "gender": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "working_hours_per_week": 40,
        "origin_country": "United-States"
    }
    result = estimator.predict(valid_input)
    assert isinstance(result, dict)
    assert "prediction" in result
    assert "conf_score" in result
    assert 0.0 <= float(result["conf_score"]) <= 1.0


def test_invalid_age_type(estimator):
    """Test that non-integer age raises InputValidationError."""
    invalid_input = {
        "age": "thirty-nine",
        "workclass": "State-gov",
        "education": "Bachelors",
        "years_of_education": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "ethinicity": "White",
        "gender": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "working_hours_per_week": 40,
        "origin_country": "United-States"
    }
    with pytest.raises(InputValidationError):
        estimator.predict(invalid_input)


def test_invalid_category(estimator):
    """Test that invalid categorical values raise InputValidationError."""
    invalid_input = {
        "age": 39,
        "workclass": "testetst",  # Invalid workclass
        "education": "Bachelors",
        "years_of_education": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "ethinicity": "White",
        "gender": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "working_hours_per_week": 40,
        "origin_country": "United-States"
    }
    with pytest.raises(InputValidationError):
        estimator.predict(invalid_input)


def test_missing_field(estimator):
    """Test that missing required fields raise InputValidationError."""
    invalid_input = {
        # "age" is missing
        "workclass": "State-gov",
        "education": "Bachelors",
        "years_of_education": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "ethinicity": "White",
        "gender": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "working_hours_per_week": 40,
        "origin_country": "United-States"
    }
    with pytest.raises(InputValidationError):
        estimator.predict(invalid_input)


def test_threshold_behavior(estimator):
    """Test that threshold logic works as expected."""
    high_prob_input = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "years_of_education": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "ethinicity": "White",
        "gender": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "working_hours_per_week": 13,
        "origin_country": "United-States"
    }
    result = estimator.predict(high_prob_input)
    assert result["prediction"] in [">50K", "<=50K"]
    assert 0.0 <= float(result["conf_score"]) <= 1.0


def test_predict_batch(estimator):
    """Test batch prediction returns correct number of results."""
    inputs = [
        {
            "age": 39,
            "workclass": "State-gov",
            "education": "Bachelors",
            "years_of_education": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "ethinicity": "White",
            "gender": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "working_hours_per_week": 40,
            "origin_country": "United-States"
        },
        {
            "age": 50,
            "workclass": "Self-emp-not-inc",
            "education": "Bachelors",
            "years_of_education": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "ethinicity": "White",
            "gender": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "working_hours_per_week": 13,
            "origin_country": "United-States"
        }
    ]
    results = estimator.predict_batch(inputs)
    assert len(results) == 2
    for res in results:
        assert "prediction" in res and "conf_score" in res
