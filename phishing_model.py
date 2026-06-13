import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# SAMPLE DATA
data = {
    "text": [
        "Click here to win money http://fake.com",
        "Your bank account is safe",
        "Urgent verify password http://hack.com",
        "Meeting at 10 AM tomorrow",
        "Free prize click now http://scam.com",
        "Project submission due tomorrow"
    ],
    "label": [
        "phishing",
        "safe",
        "phishing",
        "safe",
        "phishing",
        "safe"
    ]
}

df = pd.DataFrame(data)

# SPLIT DATA
X = df["text"]
y = df["label"]

# TEXT TO NUMBERS
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# LABEL ENCODING
y = y.map({"safe": 0, "phishing": 1})

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# ACCURACY
print("Accuracy:", accuracy_score(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print(cm)

# GRAPH
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Phishing Detection")
plt.show()

# TEST FUNCTION
def check_email(text):
    text_vec = vectorizer.transform([text])
    pred = model.predict(text_vec)[0]

    if pred == 1:
        return "⚠ Phishing Email"
    else:
        return "✅ Safe Email"

print(check_email("Urgent! click http://fake.com"))