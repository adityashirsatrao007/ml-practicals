import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# UCI Breast Cancer Wisconsin (Diagnostic) dataset.
cancer_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
cancer = pd.read_csv(cancer_url, header=None)
cancer.columns = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]

X = cancer.drop(columns=["id", "diagnosis"])
y = (cancer["diagnosis"] == "M").astype(int)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(DecisionTreeClassifier(max_depth=3), X, y, cv=kf)

print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_classifier = DecisionTreeClassifier(max_depth=1)
adaboost_classifier = AdaBoostClassifier(
    estimator=base_classifier, n_estimators=50, random_state=42
)
adaboost_classifier.fit(X_train, y_train)
y_pred = adaboost_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
