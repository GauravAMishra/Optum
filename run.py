
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from income_estimator.estimator import IncomeEstimator


income_estimator = IncomeEstimator(model_path="models/income_estimator_rf.pkl", threshold=0.9)

# with open("E:/Data Science (Self Study & Class)/Interview/optum/cases.json", "r") as f:
#     test_data = json.load(f)

with open("E:/Data Science (Self Study & Class)/Interview/optum/testcases.json", "r") as f:
    test_data = json.load(f)



for idx, case in enumerate(test_data):
    
    result = income_estimator.predict(case)
    print(result)