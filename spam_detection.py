import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv("spam.csv")
data["Label"] = data["Label"].map({"spam": 1, "ham": 0})

X = data["Message"]
y = data["Label"]

# ---------------- TRAIN MODEL ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

accuracy = model.score(X_test_vec, y_test)

print("\n📊 MODEL ACCURACY:", round(accuracy * 100, 2), "%")

# ---------------- PREDICT FUNCTION ---------------- #
def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    prob = model.predict_proba(msg_vec)[0]

    ham_prob = prob[0]
    spam_prob = prob[1]

    if spam_prob > ham_prob:
        result = "SPAM 🚨"
        color = "red"
    else:
        result = "SAFE ✅"
        color = "green"

    confidence = round(max(ham_prob, spam_prob) * 100, 2)

    # ---------------- OUTPUT ---------------- #
    print("\n📩 Message:", msg)
    print("🔍 Result:", result)
    print("📊 Confidence:", confidence, "%")

    # ---------------- PIE CHART ---------------- #
    plt.figure(figsize=(6,6))

    labels = ["Safe (Ham)", "Spam"]
    values = [ham_prob, spam_prob]
    colors = ["green", "red"]

    plt.pie(values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90)

    plt.title("📊 Spam Detection Dashboard")

    plt.show()

    # ---------------- BIG RESULT SCREEN ---------------- #
    plt.figure(figsize=(6,3))
    plt.text(0.5, 0.5, result,
             fontsize=30,
             ha='center',
             va='center',
             color=color,
             fontweight='bold')

    plt.axis("off")
    plt.show()

# ---------------- TEST ---------------- #
predict_message("Win a free iPhone now!!! Click here")
predict_message("Hey, are we meeting tomorrow for project discussion?")