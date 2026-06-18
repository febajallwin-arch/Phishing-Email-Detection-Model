"""
dataset.py — Synthetic phishing / legitimate email dataset generator
Phishing Email Detection Model

Since this project needs a labeled dataset and no external dataset is bundled,
this script generates a realistic synthetic dataset by combining template
sentences, random names, companies, suspicious / legitimate URLs, and noise.

For a real-world deployment, replace this with a public dataset such as:
  - Kaggle "Phishing Email Detection" dataset
  - The Nazario phishing corpus + Enron "ham" emails

Run directly to (re)generate data/phishing_dataset.csv:
    python dataset.py
"""

import random
import os
import pandas as pd

random.seed(42)

NAMES = [
    "John", "Sarah", "Michael", "Priya", "David", "Emma", "Carlos", "Aisha",
    "Liam", "Sophia", "Ravi", "Olivia", "James", "Mei", "Daniel", "Grace",
    "Arjun", "Laura", "Tom", "Nina",
]

PHISH_COMPANIES = [
    "PayPal", "Amazon", "Apple", "Netflix", "Chase Bank", "Wells Fargo",
    "Bank of America", "Microsoft", "Google", "DHL Express", "FedEx",
    "IRS Tax Department", "Facebook", "Instagram", "American Express",
]

LEGIT_COMPANIES = [
    "Acme Corp", "BrightPath Consulting", "Riverside Bank", "Global University",
    "Sunrise Logistics", "Greenfield Realty", "NovaTech Solutions",
    "Harbor Health Clinic", "Maple Street Bakery", "Falcon Software Inc",
    "Pinecrest Insurance", "Cedar Grove HR Team", "BlueSky Airlines",
    "Lighthouse Media", "Orchard Valley School",
]

SUSPICIOUS_DOMAINS = [
    "secure-login-verify.tk", "account-update.xyz", "paypal-security-check.info",
    "verify-now-account.ru", "billing-update-center.cn", "login-alert-secure.ga",
    "192.168.43.91", "10.0.55.21", "amaz0n-account-help.xyz",
    "appleid-confirm-now.tk", "bank-secure-verification.info",
]

LEGIT_DOMAINS = [
    "www.acmecorp.com", "portal.brightpathconsulting.com", "mail.google.com",
    "www.riversidebank.com", "intranet.globaluniversity.edu",
    "www.sunriselogistics.com", "support.novatechsolutions.com",
    "patient.harborhealthclinic.com", "orders.maplestreetbakery.com",
    "www.falconsoftware.com",
]

PHISH_TEMPLATES = [
    "Dear {name}, your {company} account has been suspended due to suspicious "
    "activity. Click here to verify your identity immediately: {url}",

    "URGENT: Your payment for {company} could not be processed. Update your "
    "billing details now at {url} or your account will be closed within 24 hours.",

    "Congratulations {name}! You have been selected to win a $1000 {company} "
    "gift card. Claim your prize now by clicking: {url}",

    "Security Alert: We detected an unusual login attempt on your {company} "
    "account. Verify your password immediately at {url} to avoid suspension.",

    "Your {company} account will be permanently deleted unless you confirm "
    "your information within 24 hours: {url}",

    "Final Notice: Your subscription to {company} has expired. Renew now to "
    "avoid losing access to your account: {url}",

    "Dear Customer, click here to claim your tax refund from {company}: {url}. "
    "Failure to respond within 48 hours will forfeit your refund.",

    "We need you to verify your bank account details urgently. Failure to "
    "respond within 24 hours will result in account suspension. Visit: {url}",

    "Your {company} package could not be delivered. Please confirm your "
    "shipping address and pay a small redelivery fee at {url}",

    "Act now! Limited time offer — your {company} account requires immediate "
    "verification at {url} to avoid permanent closure.",

    "Dear {name}, this is your final reminder from {company}. Your account "
    "access will be revoked today unless you re-confirm your password at {url}",

    "ALERT: {company} has detected unauthorized access to your account. "
    "Click {url} now to secure your account before it is locked.",
]

LEGIT_TEMPLATES = [
    "Hi {name}, thanks for signing up for {company}. Here's how to get "
    "started with your new account: {url}",

    "Hello {name}, your monthly invoice from {company} is attached. Please "
    "review it and let us know if you have any questions.",

    "Hi {name}, this is a reminder that your meeting with the {company} "
    "team is scheduled for tomorrow at 10am.",

    "Dear {name}, thank you for your recent purchase from {company}. Your "
    "order has shipped and will arrive in 3-5 business days. Track it at {url}",

    "Hi team, please find attached the quarterly report for {company}. Let "
    "me know if you have any feedback before Friday.",

    "Hello {name}, your subscription to the {company} newsletter has been "
    "confirmed. We hope you enjoy our content.",

    "Hi {name}, just checking in to see how the project is going. Let's "
    "catch up sometime this week if you're free.",

    "Dear {name}, your appointment with {company} has been confirmed for "
    "next Monday at 3pm. Reply if you need to reschedule.",

    "Hi {name}, here is the document you requested from {company}. Let me "
    "know if you need anything else.",

    "Hello, welcome to the {company} community! We're excited to have you "
    "on board. Visit {url} to explore your dashboard.",

    "Hi {name}, the {company} office will be closed on Monday for the "
    "public holiday. Normal hours resume Tuesday.",

    "Dear {name}, attached is your receipt from {company} for your records. "
    "Thanks again for your business.",
]


def random_url(suspicious: bool) -> str:
    domain = random.choice(SUSPICIOUS_DOMAINS if suspicious else LEGIT_DOMAINS)
    path = random.choice(["verify", "login", "update", "account", "confirm",
                           "secure", "dashboard", "orders", "support", ""])
    scheme = "http" if suspicious else "https"
    return f"{scheme}://{domain}/{path}".rstrip("/")


def make_phishing_email() -> str:
    template = random.choice(PHISH_TEMPLATES)
    text = template.format(
        name=random.choice(NAMES),
        company=random.choice(PHISH_COMPANIES),
        url=random_url(suspicious=True),
    )
    if random.random() < 0.4:
        text += " " + random.choice([
            "This is your final warning!!!",
            "Do not ignore this message.",
            "Your immediate action is required.",
            "ACT NOW before it's too late.",
        ])
    if random.random() < 0.3:
        text = text.replace(".", "!")
    return text


def make_legit_email() -> str:
    template = random.choice(LEGIT_TEMPLATES)
    text = template.format(
        name=random.choice(NAMES),
        company=random.choice(LEGIT_COMPANIES),
        url=random_url(suspicious=False),
    )
    if random.random() < 0.3:
        text += " " + random.choice([
            "Have a great day!",
            "Best regards.",
            "Let us know if you have questions.",
            "Thanks for your time.",
        ])
    return text


def make_hard_phishing_email() -> str:
    """Spear-phishing style: urgent wording but a normal-looking domain, no
    suspicious TLD/keyword spam — harder for the model to catch."""
    template = random.choice(PHISH_TEMPLATES)
    text = template.format(
        name=random.choice(NAMES),
        company=random.choice(PHISH_COMPANIES + LEGIT_COMPANIES),
        url=random_url(suspicious=False),
    )
    return text


def make_hard_legit_email() -> str:
    """Genuinely legitimate but urgent email (real deadline, overdue invoice)
    that uses some urgency wording — harder for the model to clear."""
    text = random.choice([
        "Hi {name}, your invoice from {company} is now overdue. Please "
        "submit payment as soon as possible to avoid a late fee.",
        "URGENT: {name}, the {company} server maintenance window starts "
        "tonight at 11pm. Please save your work and log off before then.",
        "Dear {name}, please confirm your attendance for the {company} "
        "annual review by end of day, this is time-sensitive.",
        "Hi {name}, your {company} password will expire in 3 days. You can "
        "update it any time from your account settings at {url}.",
        "Reminder: {name}, the {company} open enrollment deadline is "
        "tomorrow. Please complete your selections before midnight.",
    ]).format(
        name=random.choice(NAMES),
        company=random.choice(LEGIT_COMPANIES),
        url=random_url(suspicious=False),
    )
    return text


def _add_noise(text: str) -> str:
    """Light random noise to avoid the dataset being too cleanly separable."""
    if random.random() < 0.15:
        text = text.replace(" the ", "  the ")  # stray double space
    if random.random() < 0.1:
        text = text.lower()
    return text


def generate_dataset(n_per_class: int = 600) -> pd.DataFrame:
    rows = []
    n_hard = int(n_per_class * 0.18)  # ~18% harder, overlapping examples
    n_easy = n_per_class - n_hard

    for _ in range(n_easy):
        rows.append({"text": _add_noise(make_phishing_email()), "label": 1})
    for _ in range(n_hard):
        rows.append({"text": _add_noise(make_hard_phishing_email()), "label": 1})

    for _ in range(n_easy):
        rows.append({"text": _add_noise(make_legit_email()), "label": 0})
    for _ in range(n_hard):
        rows.append({"text": _add_noise(make_hard_legit_email()), "label": 0})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    # Simulate a small amount of real-world label noise (mislabeled samples
    # happen in real annotated datasets too) so the model isn't evaluated
    # against an unrealistically perfect, noise-free dataset.
    noise_rate = 0.04
    n_flip = int(len(df) * noise_rate)
    flip_idx = df.sample(n=n_flip, random_state=7).index
    df.loc[flip_idx, "label"] = 1 - df.loc[flip_idx, "label"]

    return df


def main():
    df = generate_dataset(n_per_class=600)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "phishing_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Generated {len(df)} emails ({df['label'].sum()} phishing, "
          f"{len(df) - df['label'].sum()} safe)")
    print(f"[+] Saved to {out_path}")


if __name__ == "__main__":
    main()
