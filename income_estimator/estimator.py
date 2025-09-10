

from __future__ import annotations
import sys 
import os

from typing import Any, Dict, List, Optional
import joblib
import pandas as pd
from pydantic import ValidationError
from .validation import RawInputModel
from .exceptions import EstimatorError, InputValidationError


class IncomeEstimator:
    """
    IncomeEstimator provides predictions from a trained RandomForest model for income classification.

    Attributes
    ----------
    model : Any
        Trained RandomForest model loaded from pickle.
    encoder : Any
        Fitted OrdinalEncoder for categorical feature transformation.
    threshold : float
        Minimum probability required to classify an input as '>50K'.
    feature_order : List[str]
        Order of features as expected by the model, inferred dynamically.

    Methods
    -------
    predict(input_data: dict) -> Dict[str, str]
        Predicts income for a single input dict.
    predict_batch(inputs: List[dict]) -> List[Dict[str, str]]
        Predicts income for a batch of inputs.
    """

    def __init__(self, model_path: str, encoder_path: str=None, threshold: float = 0.5) -> None:
        """
        Initialize IncomeEstimator with model, encoder, and threshold.

        Parameters
        ----------
        model_path : str
            Path to the trained RandomForest pickle file.
        encoder_path : str
            Path to the fitted OrdinalEncoder pickle file.
        threshold : float, default=0.5
            Minimum probability required for '>50K' classification.

        Raises
        ------
        EstimatorError
            If model or encoder fails to load.
        ValueError
            If threshold is not in [0, 1].
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1. Got {threshold}.")
        self.threshold = threshold

        self.model = self._load_model(model_path)

        if encoder_path is None:
            default_encoder_path = os.path.join("models", "ord_encoder.pkl")
            if not os.path.exists(default_encoder_path):
                raise EstimatorError(
                    f"Default encoder path {default_encoder_path} does not exist. "
                    "Please provide encoder_path explicitly."
                )
            encoder_path = default_encoder_path
            
        self.encoder = self._load_model(encoder_path)

        if hasattr(self.model, "feature_names_in_"):
            self.feature_order = list(self.model.feature_names_in_)
            
        else:
            # fallback: raise if model lacks feature_names_in_
            raise EstimatorError(
                "Model does not have 'feature_names_in_' attribute. "
                "Feature order cannot be determined dynamically."
            )

        self.positive_class_index = self._get_positive_class_index()

    def _load_model(self, path: str) -> Any:
        """Load a pickle model from disk."""
        try:
            return joblib.load(path)
        except Exception as e:
            raise EstimatorError(f"Failed to load model from {path}. Error: {str(e)}") from e

    def _validate_input(self, input_data: dict) -> RawInputModel:
        """
        Validate raw input using Pydantic model.

        Parameters
        ----------
        input_data : dict
            Raw input features from user.

        Returns
        -------
        RawInputModel
            Validated input dataclass.

        Raises
        ------
        InputValidationError
            If validation fails.
        """
        try:
            validated = RawInputModel.model_validate(input_data)
            return validated
        except ValidationError as e:
            error_msgs = "; ".join([f"{e['loc']}: {e['msg']}" for e in e.errors()])
            raise InputValidationError(f"Input validation failed: {error_msgs}")

    def _preprocess(self, validated_input: RawInputModel) -> pd.DataFrame:
        """
        Convert validated input into model-ready DataFrame.

        Parameters
        ----------
        validated_input : RawInputModel
            Pydantic validated input.

        Returns
        -------
        pd.DataFrame
            Preprocessed DataFrame ready for model inference.
        """
        # Convert dataclass to dict
        
        input_dict = validated_input.dict()
        

        # Separate categorical and numeric columns
        categorical_cols = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "ethinicity",
            "gender",
            "origin_country",
        ]
        numeric_cols = [
            "age",
            "capital_gain",
            "capital_loss",
            "working_hours_per_week",
            "years_of_education",
        ]

        # Prepare DataFrame
        df_numeric = pd.DataFrame({col: [input_dict[col]] for col in numeric_cols})
        df_numeric["working_hours_per_weekint"] = df_numeric["working_hours_per_week"]
        df_numeric = df_numeric.drop(columns=["working_hours_per_week"])

        df_categorical = pd.DataFrame({col: [input_dict[col].strip()] for col in categorical_cols})
        df_categorical_order = df_categorical[self.encoder.feature_names_in_]
        # Encode categorical columns using fitted OrdinalEncoder
        try:
            
            df_encoded = pd.DataFrame(
                self.encoder.transform(df_categorical_order.reset_index().drop(columns=["index"])),
                columns=self.encoder.feature_names_in_
            )
        except ValueError as e:
            raise InputValidationError(
                f"Categorical encoding error. Check input values: {str(e)}"
            ) from e

        # Combine numeric and encoded categorical columns
        df_final = pd.concat([df_numeric, df_encoded], axis=1)

        # Reorder columns dynamically according to model.feature_names_in_
        df_final = df_final[self.feature_order]

        return df_final

    def _get_positive_class_index(self) -> int:
        """
        Identify the index of the '>50K' class in model.classes_.

        Returns
        -------
        int
            Index of positive class.

        Raises
        ------
        EstimatorError
            If positive class cannot be determined.
        """
        classes = [str(cls).strip() for cls in self.model.classes_]
        for idx, cls in enumerate(classes):
            if cls in [">50K", ">50K."]:
                return idx
        # fallback: assume numeric classes
        if all(isinstance(c, (int, float)) for c in self.model.classes_):
            return 1
        raise EstimatorError(
            f"Could not identify positive class '>50K' in model.classes_: {self.model.classes_}"
        )

    @staticmethod
    def _format_conf_score(prob: float) -> str:
        """
        Format confidence score as string with two decimals.

        Parameters
        ----------
        prob : float
            Probability value.

        Returns
        -------
        str
            Formatted probability string.
        """
        return f"{prob:.2f}"

    def predict(self, input_data: dict) -> Dict[str, str]:
        """
        Predict income for a single input.

        Parameters
        ----------
        input_data : dict
            Raw input features from user.

        Returns
        -------
        dict
            Prediction dictionary: {"prediction": ">50K", "conf_score": "0.95"}

        Raises
        ------
        InputValidationError
            If input fails validation or encoding.
        EstimatorError
            If prediction fails.
        """
        validated = self._validate_input(input_data)
        df_preprocessed = self._preprocess(validated)

        try:
            proba = self.model.predict_proba(df_preprocessed)[0]
        except Exception as e:
            raise EstimatorError(f"Model prediction failed: {str(e)}") from e

        prob_pos = proba[self.positive_class_index]
        # print("actual prediction",self.model.predict(df_preprocessed))
        if prob_pos >= self.threshold:
            prediction = ">50K"
        else:
            prediction = "<=50K"

        return {"prediction": prediction, "conf_score": self._format_conf_score(prob_pos)}

    def predict_batch(self, inputs: List[dict]) -> List[Dict[str, str]]:
        """
        Predict income for a batch of inputs.

        Parameters
        ----------
        inputs : List[dict]
            List of raw input feature dicts.

        Returns
        -------
        List[dict]
            List of prediction dictionaries.

        Raises
        ------
        InputValidationError
            If any input fails validation or encoding.
        EstimatorError
            If prediction fails.
        """
        results = []
        for input_data in inputs:
            results.append(self.predict(input_data))
        return results
