import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Correlation Heatmap
data = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
corr = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# 2. Pairplot
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", palette="husl", diag_kind="kde")
plt.show()

# 3. Categorical Plot
tips = sns.load_dataset("tips")
plt.figure(figsize=(8, 6))
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", palette="muted")
plt.title("Average Total Bill by Day and Gender")
plt.show()

# 4. Violin Plot
plt.figure(figsize=(8, 6))
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True, palette="Set2")
plt.title("Distribution of Total Bills")
plt.show()

# 5. Joint Plot
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", cmap="Blues")
plt.show()

# 6. Boxen Plot
diamonds = sns.load_dataset("diamonds")
plt.figure(figsize=(8, 6))
sns.boxenplot(data=diamonds, x="cut", y="price", palette="coolwarm")
plt.title("Price Distribution by Cut")
plt.show()
