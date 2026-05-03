import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train():
    # ── Load data ──────────────────────────────────────────────────
    print("📂 Loading dataset...")
    df = pd.read_csv("data/diabetes.csv")
    print(f"   Shape: {df.shape}")
    print(f"   Diabetic: {df['Outcome'].sum()} | Non-diabetic: {(df['Outcome']==0).sum()}")

    # ── Clean impossible zeros ─────────────────────────────────────
    print("\n🧹 Cleaning data...")
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    for col in zero_cols:
        zeros = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())        
        print(f"   {col}: replaced {zeros} zeros with median")

    # ── Split ──────────────────────────────────────────────────────
    print("\n✂️  Splitting data...")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── Build pipeline ─────────────────────────────────────────────
    print("\n🔧 Building pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # ── Cross validation ───────────────────────────────────────────
    print("\n🔁 Running cross-validation...")
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=5, scoring="accuracy"
    )
    print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Train ──────────────────────────────────────────────────────
    print("\n🚀 Training model...")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────
    print("\n📊 Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   Test Accuracy: {acc:.3f}")
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=["No Diabetes", "Diabetes"]
    ))

    # ── Confusion matrix ───────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Diabetes", "Diabetes"],
        yticklabels=["No Diabetes", "Diabetes"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png")
    print("✅ Confusion matrix saved → data/confusion_matrix.png")

    # ── Feature importance ─────────────────────────────────────────
    importances = pipeline.named_steps["model"].feature_importances_
    feat_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.title("Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png")
    print("✅ Feature importance saved → data/feature_importance.png")

    # ── Save model ─────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    with open("models/diabetes_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    print("✅ Model saved → models/diabetes_model.pkl")

if __name__ == "__main__":
    train()