import pandas as pd
import matplotlib.pyplot as plt

# Load the results
df = pd.read_csv('results.csv')
plt.figure(figsize=(8, 6))
plt.hist(df['mean'], bins=50, edgecolor='black')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.title('Distribution of Mean Temperatures')
plt.tight_layout()
plt.savefig('hist_mean_temperature.png')
plt.close()
print("Saved histogram to hist_mean_temperature.png")

# 2. Average mean temperature
avg_mean = df['mean'].mean()

# 3. Average temperature standard deviation
avg_std = df['std'].mean()

# 4. Count of buildings with ≥50% area above 18°C
count_above_18 = (df['pct>18'] >= 50).sum()

# 5. Count of buildings with ≥50% area below 15°C
count_below_15 = (df['pct<15'] >= 50).sum()

# Print summary
print(f"Average mean temperature: {avg_mean:.2f} °C")
print(f"Average temperature standard deviation: {avg_std:.2f} °C")
print(f"Buildings with ≥50% area above 18°C: {count_above_18}")
print(f"Buildings with ≥50% area below 15°C: {count_below_15}")
