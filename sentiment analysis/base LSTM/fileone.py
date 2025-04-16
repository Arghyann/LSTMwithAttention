import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Set a modern aesthetic style
sns.set_style("darkgrid")
plt.rcParams.update({'font.family': 'sans-serif'})

df=pd.read_csv("IMDB Dataset.csv")
print(df.columns)
sentiment_counts = df.iloc[:,1].value_counts()

plt.figure(figsize=(10, 6))

# Custom color palette
colors = ["#2ecc71", "#e74c3c"]  # Modern green and red
ax = sentiment_counts.plot(
    kind="bar", 
    color=colors,
    width=0.6,
    edgecolor='black',
    linewidth=1.5
)

# Add data labels on top of bars
for i, count in enumerate(sentiment_counts):
    ax.text(i, count + 100, f'{count:,}', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Enhance chart appearance
plt.xlabel("Sentiment", fontsize=14, fontweight='bold')
plt.ylabel("Count", fontsize=14, fontweight='bold')
plt.title("Movie Review Sentiment Analysis", fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=0, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12)

# Add a subtle grid for readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a background color to the plot area
ax.set_facecolor("#f8f9fa")

# Add context with a subtitle
plt.figtext(0.5, 0.01, "Based on IMDB Movie Reviews Dataset", 
            ha="center", fontsize=10, fontstyle='italic')

plt.tight_layout()
plt.show()