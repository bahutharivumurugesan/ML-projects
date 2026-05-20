import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("social_media.csv")

# Professional Style
sns.set_style("darkgrid")
sns.set_palette("bright")

# Bigger overall figure style
plt.rcParams["figure.figsize"] = (10, 6)

# ====================================
# 1. Followers vs Likes
# ====================================

plt.figure()

sns.scatterplot(
    data=df,
    x="Followers",
    y="Likes",
    hue="Platform",
    size="Daily_Usage_Hours",
    sizes=(100, 400),
    palette="Set2"
)

plt.title(
    "Followers vs Likes Analysis",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Followers", fontsize=12)
plt.ylabel("Likes", fontsize=12)

plt.tight_layout()
plt.show()

# ====================================
# 2. Platform Popularity
# ====================================

plt.figure()

sns.countplot(
    data=df,
    x="Platform",
    hue="Platform",
    palette="viridis",
    legend=False
)

plt.title(
    "Most Popular Social Media Platforms",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Platform", fontsize=12)
plt.ylabel("Count", fontsize=12)

plt.tight_layout()
plt.show()

# ====================================
# 3. Daily Usage Distribution
# ====================================

plt.figure()

sns.histplot(
    df["Daily_Usage_Hours"],
    bins=6,
    kde=True,
    color="cyan"
)

plt.title(
    "Daily Social Media Usage",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Usage Hours", fontsize=12)
plt.ylabel("Frequency", fontsize=12)

plt.tight_layout()
plt.show()

# ====================================
# 4. Correlation Heatmap
# ====================================

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include='number')

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=1,
    linecolor='white'
)

plt.title(
    "Correlation Between Features",
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout()
plt.show()

# ====================================
# 5. Posts Per Week Analysis
# ====================================

plt.figure()

sns.barplot(
    data=df,
    x="Platform",
    y="Posts_Per_Week",
    hue="Platform",
    palette="magma",
    legend=False
)

plt.title(
    "Posting Activity Per Platform",
    fontsize=18,
    fontweight='bold'
)

plt.xlabel("Platform", fontsize=12)
plt.ylabel("Posts Per Week", fontsize=12)

plt.tight_layout()
plt.show()