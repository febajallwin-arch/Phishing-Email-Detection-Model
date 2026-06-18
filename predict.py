"""
predict.py — Classify new emails as "Phishing" or "Safe"
Loads the trained model from model/phishing_model.joblib

Usage:
    python predict.py --text "Your account has been suspended, click here..."
    python predict.py --file sample_email.txt
    python predict.py --csv emails.csv --output predictions.csv
    python predict.py            (interactive mode)
"""

import argparse
import os
import sys
import joblib
import pandas as pd

MODEL_PATH = os.path.join("model", "phishing_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[!] No trained model found at {MODEL_PATH}")
        print("    Run 'python train_model.py' first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def classify(model, text: str):
    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred] * 100
    return label, confidence


def print_result(text: str, label: str, confidence: float):
    preview = text.strip().replace("\n", " ")
    if len(preview) > 90:
        preview = preview[:90] + "..."
    tag = "[PHISHING]" if label == "Phishing" else "[SAFE]"
    print(f"  {tag:<12} ({confidence:5.1f}% confidence)  \"{preview}\"")


def run_single(model, text: str):
    label, confidence = classify(model, text)
    print()
    print_result(text, label, confidence)
    print()


def run_csv(model, csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    if "text" not in df.columns:
        print("[!] CSV must contain a 'text' column.")
        sys.exit(1)

    labels, confidences = [], []
    for t in df["text"].astype(str):
        label, confidence = classify(model, t)
        labels.append(label)
        confidences.append(round(confidence, 1))

    df["prediction"] = labels
    df["confidence_pct"] = confidences
    df.to_csv(output_path, index=False)
    print(f"[+] Classified {len(df)} emails -> {output_path}")
    print(f"    Phishing: {labels.count('Phishing')}   Safe: {labels.count('Safe')}")


def run_interactive(model):
    print("Phishing Email Detector — interactive mode (type 'quit' to exit)\n")
    while True:
        try:
            text = input("Paste email text > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        label, confidence = classify(model, text)
        print_result(text, label, confidence)
        print()


def build_parser():
    p = argparse.ArgumentParser(
        description="Classify an email as Phishing or Safe using the trained model."
    )
    p.add_argument("--text", help="Email text to classify")
    p.add_argument("--file", help="Path to a .txt file containing the email body")
    p.add_argument("--csv", help="Path to a CSV file with a 'text' column for batch classification")
    p.add_argument("--output", default="predictions.csv", help="Output CSV path for --csv mode")
    return p


def main():
    args = build_parser().parse_args()
    model = load_model()

    if args.csv:
        run_csv(model, args.csv, args.output)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
        run_single(model, text)
    elif args.text:
        run_single(model, args.text)
    else:
        run_interactive(model)


if __name__ == "__main__":
    main()
