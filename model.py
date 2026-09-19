import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/placement_data.csv")

print("==========================================")
print("COLLEGE PLACEMENT PREDICTION")
print("==========================================")

print("\nDataset loaded successfully!")
print("Number of students:", len(df))

print("\nColumns in dataset:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())


# ==========================================
# 2. TARGET COLUMN
# ==========================================

target = "Placement"

print("\nTarget column:", target)


# ==========================================
# 3. SELECT FEATURES
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

print("\nFeatures used for prediction:")
for feature in features:
    print("-", feature)


# ==========================================
# 4. PREPARE DATA
# ==========================================

X = df[features]
y = df[target]


# ==========================================
# 5. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n==========================================")
print("DATA SPLIT")
print("==========================================")

print("Training students:", len(X_train))
print("Testing students:", len(X_test))


# ==========================================
# 6. CREATE MACHINE LEARNING MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Machine Learning model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 9. MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.to_string(index=False))


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "placement_model.pkl"
)

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY")
print("==========================================")

print("File created: placement_model.pkl")

print("\nStep 3 - Machine Learning Prediction")
print("completed successfully!")