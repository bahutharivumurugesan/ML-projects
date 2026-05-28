import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("loan_data.csv")

# Inputs
X = df[["Income", "CreditScore", "Loan"]]

# Output
y = df["Approved"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# New customer prediction
new_customer = [[42000, 710, 160000]]

result = model.predict(new_customer)

# Output
if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")