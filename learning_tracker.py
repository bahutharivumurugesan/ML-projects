import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Create Dataset
# -----------------------------
data = {
    "Topic": [
        "NumPy",
        "Pandas",
        "Matplotlib",
        "Seaborn",
        "EDA",
        "Regression",
        "Classification",
        "OpenCV"
    ],

    "Hours_Practiced": [
        12,
        18,
        15,
        14,
        10,
        8,
        7,
        5
    ],

    "Projects_Done": [
        2,
        3,
        2,
        2,
        1,
        1,
        1,
        0
    ],

    "Confidence_Level": [
        7,
        8,
        7,
        8,
        6,
        5,
        5,
        3
    ]
}

learning = pd.DataFrame(data)

# -----------------------------
# Dataset Info
# -----------------------------
print(learning.head())

print("\nDataset Shape:")
print(learning.shape)

# -----------------------------
# Bar Plot
# -----------------------------
plt.figure(figsize=(8,5))

sns.barplot(
    x="Topic",
    y="Hours_Practiced",
    data=learning,
    palette="viridis"
)

plt.title("Hours Practiced Per Topic")

plt.xticks(rotation=20)

plt.show()

# -----------------------------
# Scatter Plot
# -----------------------------
plt.figure(figsize=(6,4))

sns.scatterplot(
    x="Hours_Practiced",
    y="Confidence_Level",
    data=learning,
    s=120
)

plt.title("Practice vs Confidence")

plt.show()

# -----------------------------
# Histogram
# -----------------------------
plt.figure(figsize=(6,4))

sns.histplot(
    data=learning,
    x="Confidence_Level",
    kde=True
)

plt.title("Confidence Distribution")

plt.show()

# -----------------------------
# Box Plot
# -----------------------------
plt.figure(figsize=(6,4))

sns.boxplot(
    y="Projects_Done",
    data=learning
)

plt.title("Projects Done Analysis")

plt.show()

# -----------------------------
# Heatmap
# -----------------------------
plt.figure(figsize=(6,4))

corr = learning[
    ["Hours_Practiced",
     "Projects_Done",
     "Confidence_Level"]
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)

plt.title("Learning Correlation")

plt.show()