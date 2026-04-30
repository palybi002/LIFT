import pandas as pd
df = pd.read_csv('ablation_results.csv')
print("--- First 25 lines of ablation_results.csv ---")
print(df.head(25))
print("\n--- Unique AblationType for AirQuality ---")
print(df[df['Dataset'] == 'AirQuality']['AblationType'].unique())
print("\n--- Unique AblationType for Weather ---")
print(df[df['Dataset'] == 'Weather']['AblationType'].unique())
