"""
Custom exceptions for the IncomeEstimator module.
"""

from typing import Any, Optional


class EstimatorError(Exception):
    """
    Raised when there is a general error in the IncomeEstimator,
    such as model loading failure or prediction failure.
    """
    pass


class InputValidationError(Exception):
    """
    Raised when user input fails validation or encoding.
    Examples:
        - Incorrect type (e.g., age="thirty-nine")
        - Categorical value not allowed
        - Missing required fields
    """
    pass
