
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from train import load_data, build_model


# Load dataset
X, y = load_data()

print("Total examples:", len(X))
print("Total intents:", len(set(y)))


# Display dataset distribution
print("\nExamples per intent:")

for label in sorted(set(y)):
    print(f"{label}: {y.count(label)}")


# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42
)


# Train model
model = build_model()
model.fit(X_train, y_train)


# Test predictions
pred = model.predict(X_test)


# Evaluation metrics
accuracy = accuracy_score(y_test, pred)
macro_f1 = f1_score(y_test, pred, average="macro")
weighted_f1 = f1_score(y_test, pred, average="weighted")


print("\n========== TEST RESULTS ==========")

print(f"Test accuracy: {accuracy:.3f}")
print(f"Macro F1-score: {macro_f1:.3f}")
print(f"Weighted F1-score: {weighted_f1:.3f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


# 5-fold stratified cross-validation
class_counts = np.unique(y, return_counts=True)[1]
n_splits = min(5, class_counts.min())

cv = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    build_model(),
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n========== CROSS-VALIDATION ==========")

print("Fold accuracies:", np.round(cv_scores, 3))
print(f"Mean accuracy: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")


# Confusion matrix
labels = sorted(set(y))

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels
)

os.makedirs("charts", exist_ok=True)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Intent")
plt.ylabel("Actual Intent")
plt.title("TechBuddy Intent Classification Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "charts/confusion_matrix.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved to charts/confusion_matrix.png")


# Misclassified examples
print("\n========== MISCLASSIFIED EXAMPLES ==========")

for text, actual, predicted in zip(X_test, y_test, pred):

    if actual != predicted:
        print(
            f"'{text}' | "
            f"Actual: {actual} | "
            f"Predicted: {predicted}"
        )


# Save evaluation summary
os.makedirs("docs", exist_ok=True)

with open("docs/evaluation_results.txt", "w", encoding="utf-8") as f:

    f.write(f"Test accuracy: {accuracy:.3f}\n")
    f.write(f"Macro F1-score: {macro_f1:.3f}\n")
    f.write(f"Weighted F1-score: {weighted_f1:.3f}\n")
    f.write(f"CV scores: {np.round(cv_scores, 3)}\n")
    f.write(f"CV mean: {cv_scores.mean():.3f}\n")
    f.write(f"CV standard deviation: {cv_scores.std():.3f}\n")

print("\nEvaluation results saved to docs/evaluation_results.txt")
