import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Generate sample data - non-linear but monotonic relationship
np.random.seed(42)
x = np.random.uniform(0, 10, 100)
y = x**2 + np.random.normal(0, 10, 100)

# Create Dataframe
df = pd.DataFrame({'feature': x, 'target': y})

# Calculate Spearman correaltion
spearman_corr, p_value = spearmanr(df['feature'], df['target'])

print(f"Spearman's rank correlation: {spearman_corr:.4f}")
print(f"P-value: {p_value:.4f}")

# Demonstrate comparison with Pearson correlation
from scipy.stats import pearsonr
pearson_corr, p_value_pearson = pearsonr(df['feature'], df['target'])
print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"P-value: {p_value_pearson:.4f}")

# Compare correlations across multiple features
# Create a dataset with multiple features
df_multi = pd.DataFrame({
    'linear': x,
    'quadratic': x**2,
    'exponential': np.exp(x/5),
    'random': np.random.normal(0, 1, 100),
    'target': y
})

# Create correlation matrix
corr_matrix = df_multi.corr(method='spearman')

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Spearman Correlation Matrix')
plt.show()