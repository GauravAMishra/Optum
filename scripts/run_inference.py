import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
from income_estimator.estimator import IncomeEstimator
from income_estimator.exceptions import EstimatorError, InputValidationError


def main(model_path: str, encoder_path: str, testcases_path: str, threshold: float = 0.9) -> None:
    """
    Load the model and encoder, read test cases from JSON, and print predictions.

    Parameters
    ----------
    model_path : str
        Path to the trained RandomForest pickle file.
    encoder_path : str
        Path to the fitted OrdinalEncoder pickle file.
    testcases_path : str
        Path to the JSON file containing test cases.
    threshold : float, default=0.9
        Minimum probability required for '>50K' classification.
    """
    # Initialize the estimator
    try:
        income_estimator = IncomeEstimator(model_path=model_path, encoder_path=encoder_path, threshold=threshold)
    except Exception as e:
        print(f"Failed to initialize IncomeEstimator: {e}")
        return

    # Load test cases
    try:
        with open(testcases_path, "r") as f:
            testcases = json.load(f)
    except Exception as e:
        print(f"Failed to load test cases from {testcases_path}: {e}")
        return

    # Predict and print results
    for idx, testcase in enumerate(testcases):
        try:
            result = income_estimator.predict(testcase)
            print(f"Test case {idx}: {result}")
        except InputValidationError as ve:
            print(f"Test case {idx} input validation error: {ve}")
        except EstimatorError as ee:
            print(f"Test case {idx} prediction error: {ee}")
        except Exception as e:
            print(f"Test case {idx} unexpected error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IncomeEstimator on JSON test cases.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to RandomForest model pickle")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to OrdinalEncoder pickle")
    parser.add_argument("--testcases_path", type=str, required=True, help="Path to testcases JSON file")
    parser.add_argument("--threshold", type=float, default=0.9, help="Probability threshold for '>50K'")

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        encoder_path=args.encoder_path,
        testcases_path=args.testcases_path,
        threshold=args.threshold,
    )
