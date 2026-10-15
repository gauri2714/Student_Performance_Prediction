
# STUDENT PERFORMANCE PREDICTION
# Predictive Analytics Mini Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)

# 1. Load dataset
import os

folder = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(folder, "student-mat.csv"), sep=";")

print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())
print("\nMissing Values:", df.isnull().sum().sum())

# 2. Visualize student grades
plt.figure(figsize=(7, 4))
sns.histplot(df["G3"], bins=10)
plt.title("Distribution of Student Final Grades")
plt.xlabel("Final Grade")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="studytime")
plt.title("Students by Study Time")
plt.tight_layout()
plt.show()

# 3. Create target variable
# 1 = At risk (final grade below 10)
# 0 = Not at risk (final grade 10 or above)

df["at_risk"] = (df["G3"] < 10).astype(int)

# Exclude grade columns to avoid using the answer as input
X = df.drop(columns=["G1", "G2", "G3", "at_risk"])
y = df["at_risk"]

# Convert categorical data into numerical columns
X = pd.get_dummies(X, drop_first=True)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 5. Logistic Regression model
model1 = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

print("\n--- LOGISTIC REGRESSION ---")
print("Accuracy:", round(accuracy_score(y_test, pred1), 3))
print("F1 Score:", round(f1_score(y_test, pred1), 3))
print(classification_report(y_test, pred1, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred1,
    display_labels=["Not at risk", "At risk"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.show()

# 6. Random Forest model
model2 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

print("\n--- RANDOM FOREST ---")
print("Accuracy:", round(accuracy_score(y_test, pred2), 3))
print("F1 Score:", round(f1_score(y_test, pred2), 3))
print(classification_report(y_test, pred2, zero_division=0))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred2,
    display_labels=["Not at risk", "At risk"]
)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# 7. Compare the two models
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_test, pred1),
        accuracy_score(y_test, pred2)
    ],
    "F1 Score": [
        f1_score(y_test, pred1),
        f1_score(y_test, pred2)
    ]
})

print("\n--- MODEL COMPARISON ---")
print(comparison.round(3).to_string(index=False))

comparison.plot(
    x="Model",
    y=["Accuracy", "F1 Score"],
    kind="bar",
    ylim=(0, 1),
    figsize=(8, 4)
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nProject completed successfully!")
