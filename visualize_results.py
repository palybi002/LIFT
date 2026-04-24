import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime

# Get current timestamp for unique filenames
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Ensure plots directory exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load data
try:
    df = pd.read_csv('comparison_results.csv')
except FileNotFoundError:
    print("Error: comparison_results.csv not found.")
    exit()

# Filter valid data
df = df.dropna(subset=['MSE', 'MAE'])

if df.empty:
    print("No valid data found in comparison_results.csv (all MSE/MAE are empty).")
    exit()

# Get unique datasets
datasets = df['Dataset'].unique()

for dataset in datasets:
    print(f"Generating plots for {dataset}...")
    dataset_df = df[df['Dataset'] == dataset]
    
    # Aggregate if multiple entries per model (take mean)
    grouped = dataset_df.groupby('Model')[['MSE', 'MAE', 'TrainTime']].mean()
    
    # 1. MSE Comparison
    plt.figure(figsize=(10, 6))
    grouped['MSE'].plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'MSE Comparison - {dataset}')
    plt.ylabel('MSE')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_MSE_comparison_{timestamp}.png')
    plt.close()

    # 2. MAE Comparison
    plt.figure(figsize=(10, 6))
    grouped['MAE'].plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title(f'MAE Comparison - {dataset}')
    plt.ylabel('MAE')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_MAE_comparison_{timestamp}.png')
    plt.close()

    # 3. Training Time Comparison
    if 'TrainTime' in grouped.columns and grouped['TrainTime'].notna().any():
        plt.figure(figsize=(10, 6))
        grouped['TrainTime'].plot(kind='bar', color='salmon', edgecolor='black')
        plt.title(f'Training Time Comparison - {dataset}')
        plt.ylabel('Time (s)')
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/{dataset}_Time_comparison_{timestamp}.png')
        plt.close()

print(f"Plots generated in 'plots/' directory with timestamp {timestamp}.")

