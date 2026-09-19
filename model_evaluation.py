import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/placement_data.csv")

print("==========================================")
print("MODEL EVALUATION & IMPROVEMENT")
print("==========================================")

print("\nDataset:")
print("Students:", len(df))


# ==========================================
# 2. FEATURES
# ==========================================

features = [
    "CGPA",
    "Tenth_Percentage",
    "Twelfth_Percentage",
    "Backlogs",
    "Internships",
    "Projects",
    "Certifications",
    "Aptitude_Score",
    "Communication_Score",
    "DSA_Score",
    "Python",
    "SQL",
    "Java",
    "Machine_Learning",
    "Web_Development",
    "Excel"
]

target = "Placement"


# ==========================================
# 3. X AND Y
# ==========================================

X = df[features]
y = df[target]


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
}


# ==========================================
# 6. TRAIN & EVALUATE MODELS
# ==========================================

results = []

trained_models = {}

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    # 5-fold cross validation
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_accuracy = cv_scores.mean()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "CV_Accuracy": cv_accuracy
    })

    trained_models[name] = model


# ==========================================
# 7. RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n==========================================")
print("EVALUATION RESULTS")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ==========================================
# 8. BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n==========================================")
print("SELECTED MODEL")
print("==========================================")

print(
    "Model selected based on highest F1-score:"
)

print(best_model_name)


# ==========================================
# 9. DETAILED EVALUATION
# ==========================================

best_prediction = best_model.predict(
    X_test
)

best_probability = best_model.predict_proba(
    X_test
)[:, 1]

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        best_prediction,
        zero_division=0
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    best_prediction
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=150
)

plt.close()


# ==========================================
# 11. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ==========================================
# 12. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    "placement_model.pkl"
)


# ==========================================
# 13. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("FINAL MODEL SUMMARY")
print("==========================================")

print(
    "Selected Model:",
    best_model_name
)

print(
    f"Accuracy: {accuracy_score(y_test, best_prediction) * 100:.2f}%"
)

print(
    f"Precision: {precision_score(y_test, best_prediction, zero_division=0) * 100:.2f}%"
)

print(
    f"Recall: {recall_score(y_test, best_prediction, zero_division=0) * 100:.2f}%"
)

print(
    f"F1 Score: {f1_score(y_test, best_prediction, zero_division=0) * 100:.2f}%"
)

print(
    f"ROC-AUC: {roc_auc_score(y_test, best_probability):.3f}"
)

print("\nFiles created:")
print("1. placement_model.pkl")
print("2. model_comparison.csv")
print("3. confusion_matrix.png")

print("\nStep 6 completed successfully!")