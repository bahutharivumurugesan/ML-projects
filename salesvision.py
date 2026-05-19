import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv("sales_data.csv")

print("\n📊 SALES DATA")
print(data)

# ---------------- INPUT & OUTPUT ---------------- #
X = data[["Month"]]
y = data["Sales"]

# ---------------- MODEL ---------------- #
model = LinearRegression()
model.fit(X, y)

# ---------------- PREDICTION ---------------- #
prediction = model.predict([[11]])

print("\n📈 Predicted Sales for Month 11:")
print(round(prediction[0], 2))

# ---------------- GRAPH ---------------- #
plt.figure(figsize=(8,5))

plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.title("📊 SalesVision - Sales Prediction")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.show()