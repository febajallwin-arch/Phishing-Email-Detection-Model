"""
train_model.py — Train and evaluate the Phishing Email Detection Model
Scikit-learn pipeline: TF-IDF + handcrafted features -> RandomForestClassifier

Run:
    python train_model.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / non-GUI environments
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from features import build_pipeline_features
from dataset import generate_dataset

DATA_PATH = os.path.join("data", "phishing_dataset.csv")
MODEL_PATH = os.path.join("model", "phishing_model.joblib")
CM_PATH = os.path.join("output", "confusion_matrix.png")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("[*] No dataset found — generating synthetic dataset...")
        df = generate_dataset(n_per_class=600)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"[+] Generated and saved {len(df)} emails to {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"[+] Loaded {len(df)} emails from {DATA_PATH}")
    return df


def build_model() -> Pipeline:
    return Pipeline([
        ("features", build_pipeline_features()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[*] Training on {len(X_train)} emails, testing on {len(X_test)} emails")

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])

    print("\n" + "=" * 60)
    print(f"  ACCURACY: {acc * 100:.2f}%")
    print("=" * 60)
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Predicted Safe   Predicted Phishing")
    print(f"  Actual Safe        {cm[0][0]:<10}      {cm[0][1]:<10}")
    print(f"  Actual Phishing    {cm[1][0]:<10}      {cm[1][1]:<10}")
    print("\nClassification Report:")
    print(report)

    # Save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    print(f"\n[+] Confusion matrix saved to {CM_PATH}")

    # Save trained pipeline (features + classifier together)
    joblib.dump(model, MODEL_PATH)
    print(f"[+] Trained model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
