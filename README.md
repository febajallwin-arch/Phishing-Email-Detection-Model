# 🛡️ Phishing Email Detection Model

A machine learning model built with **Scikit-learn** that classifies emails
as **Phishing** or **Safe** based on textual content and URL features.

---

## How it works

The model combines two feature sources into a single Scikit-learn pipeline:

1. **TF-IDF text features** — word and bigram frequencies across the email body.
2. **Handcrafted features** — number of links, suspicious keywords ("verify",
   "suspended", "urgent", "click here", etc.), IP-based URLs, suspicious
   top-level domains (.tk, .xyz, .ru, .info), exclamation marks, dollar signs,
   generic greetings ("Dear Customer"), and more.

These are combined with `FeatureUnion` and fed into a `RandomForestClassifier`.
The whole pipeline (feature extraction + classifier) is saved as a single
file, so predicting on new emails is a one-line load.

### Dataset

No labeled phishing dataset is bundled with this project, so `dataset.py`
**generates a synthetic dataset** of ~1,150 emails (phishing + legitimate)
from realistic templates, with some intentionally harder/overlapping
examples and a small amount of label noise — so the model isn't evaluated
against an unrealistically perfect dataset.

> For a real-world deployment, swap in an actual dataset, such as Kaggle's
> "Phishing Email Detection" dataset or the Nazario phishing corpus combined
> with the Enron "ham" email set. Just keep the same CSV format: a `text`
> column and a `label` column (1 = phishing, 0 = safe).

---

## Project Structure

```
phishing_email_detector/
├── dataset.py        — Generates the synthetic training dataset
├── features.py        — TF-IDF + handcrafted feature extraction
├── train_model.py     — Trains the model, prints accuracy + confusion matrix
├── predict.py          — Classifies new emails (single, file, batch CSV, interactive)
├── requirements.txt
├── data/
│   └── phishing_dataset.csv     (created by dataset.py / train_model.py)
├── model/
│   └── phishing_model.joblib    (created by train_model.py)
└── output/
    └── confusion_matrix.png     (created by train_model.py)
```

---

## Setup

**1. Install Python 3.8+** if you don't already have it.

**2. Install dependencies** (from inside the project folder):
```bash
pip install -r requirements.txt
```

---

## Step-by-Step: How to Run

### Step 1 — Train the model
```bash
python train_model.py
```
This automatically generates the synthetic dataset (if it doesn't already
exist in `data/`), trains the model, and prints:
- Overall **accuracy**
- A **confusion matrix** (printed in the terminal and saved as an image)
- A full **classification report** (precision / recall / F1 per class)

It also saves the trained model to `model/phishing_model.joblib` and the
confusion matrix plot to `output/confusion_matrix.png`.

Expected output looks like:
```
============================================================
  ACCURACY: 96.14%
============================================================

Confusion Matrix (rows=actual, cols=predicted):
                 Predicted Safe   Predicted Phishing
  Actual Safe        107             6
  Actual Phishing    3               117
```

### Step 2 — Classify a single email
```bash
python predict.py --text "Dear customer, your account has been suspended. Click here to verify: http://secure-login-verify.tk"
```
Output:
```
  [PHISHING]   ( 86.3% confidence)  "Dear customer, your account has been suspended..."
```

### Step 3 — Classify an email from a text file
```bash
python predict.py --file sample_email.txt
```

### Step 4 — Classify many emails at once (batch CSV)
Create a CSV with a `text` column, e.g. `emails.csv`:
```csv
text
"Your PayPal account has been limited. Verify now at http://paypal-verify-now.tk"
"Hi team, the meeting has been moved to 2pm Thursday."
```
Then run:
```bash
python predict.py --csv emails.csv --output predictions.csv
```
This adds `prediction` and `confidence_pct` columns and saves the result
to `predictions.csv`.

### Step 5 — Interactive mode
Just run with no arguments to paste in emails one at a time:
```bash
python predict.py
```

### (Optional) Step 6 — Regenerate the dataset with more samples
```bash
python dataset.py
```
Edit `n_per_class` inside `dataset.py`'s `main()` function to generate more
or fewer emails per class, then re-run `train_model.py`.

---

## Re-training after changes

If you edit `features.py` (e.g. add new handcrafted features) or change the
classifier in `train_model.py`, just re-run:
```bash
python train_model.py
```
This overwrites `model/phishing_model.joblib` with the newly trained version.

---

## Notes & Limitations

- The bundled dataset is **synthetic** — it's built for learning and
  demonstrating the full pipeline (feature engineering → training →
  evaluation → inference), not for production phishing detection.
- Real phishing emails are far more varied; for production use, retrain on
  a large real-world labeled dataset and consider adding features like
  sender domain reputation, SPF/DKIM/DMARC results, and HTML structure
  analysis (e.g. mismatched anchor text vs. actual link).
- `RandomForestClassifier` was chosen for a good balance of accuracy and
  interpretability, but the same pipeline works with `LogisticRegression`,
  `SVC`, or gradient boosting models — just swap the classifier in
  `train_model.py`'s `build_model()` function.
