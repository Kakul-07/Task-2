import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import preprocessor

df = pd.read_csv("Dataset/dataset.csv")

df["Security_Type"] = df["Security_Type"].replace("Indriect", "Indirect")

X = df.drop(columns=[
    "ID",
    "year",
    "Status",
    "Security_Type",
    "Interest_rate_spread",
    "Upfront_charges",
    "rate_of_interest",
    "construction_type",
    "Secured_by",
    "credit_type"
])
y = df["Status"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(C=1,solver="liblinear", class_weight="balanced", max_iter=1000))])

decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(max_depth=10, min_samples_split=10, class_weight="balanced", random_state=42))])

logistic_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)


y_pred_lr = logistic_model.predict(X_test)
y_prob_lr = logistic_model.predict_proba(X_test)[:, 1]
y_pred_dt = decision_tree_model.predict(X_test)
y_prob_dt = decision_tree_model.predict_proba(X_test)[:, 1]

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1:", f1_score(y_test, y_pred_dt))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_dt))


joblib.dump(decision_tree_model, "models/best_model.pkl")
print("\nFinal model saved successfully.")