# Loan Default Prediction

## Project Overview
This project predicts whether a loan is likely to default using machine learning.
The project uses the Loan Default Dataset containing loan, applicant and financial information.

## Dataset
- Rows: 148,670
- Columns: 34
- Target: Status
- 0 = No Default
- 1 = Default

## Project Steps
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Hyperparameter Tuning
8. Model Saving
9. Streamlit Deployment

## Data Cleaning
- Checked missing values
- Handled categorical missing values
- Handled numerical missing values
- Removed duplicate records
- Fixed category spelling
- Checked outliers
- Removed features showing possible target leakage

## Models Used
### Logistic Regression
Used as a baseline classification model.

### Decision Tree
Used to capture non-linear relationships between loan features.

## Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

## Final Results
|       Model            | Accuracy | Precision | Recall | F1     | ROC-AUC |
| Logistic Regression    | 70.56%   | 43.44%    | 64.42% | 51.89% | 74.69% |
| Decision Tree | 85.77% | 72.11%   | 68.89%    | 70.46% | 86.47% |

## Hyperparameter Tuning
### Logistic Regression
- C = 1
- Solver = liblinear
### Decision Tree
- Max Depth = 2
- Minimum Samples Split = 10
## Final Model
The tuned Decision Tree model was saved as:
`models/best_model.pkl`

## Deployment
The model was deployed using Streamlit.
Run the application:
```bash
streamlit run app.py