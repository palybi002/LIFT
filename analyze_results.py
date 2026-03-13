import os
import re
import pandas as pd
import glob

def parse_log_file(filepath):
    """Parses a single log file to extract metrics."""
    metrics = {
        'Model': 'Unknown',
        'Dataset': 'Unknown',
        'Features': 'Unknown',
        'MSE': None,
        'MAE': None,
        'Params': None,
        'TrainTime': None
    }
    
    # Extract info from filename: Model_Dataset_Features.log
    filename = os.path.basename(filepath)
    parts = filename.replace('.log', '').split('_')
    if len(parts) >= 3:
        metrics['Model'] = parts[0]
        metrics['Dataset'] = parts[1]
        metrics['Features'] = parts[2]
        
    with open(filepath, 'r') as f:
        content = f.read()
        
        # Extract MSE/MAE
        # Pattern: mse:0.1234, mae:0.5678
        match_metrics = re.search(r'mse:([0-9.]+), mae:([0-9.]+)', content)
        if match_metrics:
            metrics['MSE'] = float(match_metrics.group(1))
            metrics['MAE'] = float(match_metrics.group(2))
            
        # Extract Params
        # Pattern: Number of Params: 12345
        match_params = re.search(r'Number of Params: ([0-9]+)', content)
        if match_params:
            metrics['Params'] = int(match_params.group(1))
            
        # Extract Training Time (Approximate - average or total)
        # Pattern: Epoch: 1 cost time: 1.234
        times = re.findall(r'cov time: ([0-9.]+)|cost time: ([0-9.]+)', content)
        if times:
            # Flatten and filter empty
            valid_times = [float(t[0] or t[1]) for t in times]
            if valid_times:
                metrics['TrainTime'] = sum(valid_times) / len(valid_times) # Avg epoch time

    return metrics

def main():
    log_files = glob.glob('logs/*.log')
    results = []
    
    for log_file in log_files:
        try:
            res = parse_log_file(log_file)
            results.append(res)
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")
            
    if not results:
        print("No results found in logs/ directory.")
        return

    df = pd.DataFrame(results)
    
    # Sort for better viewing
    df = df.sort_values(by=['Dataset', 'MSE'])
    
    print("\n" + "="*80)
    print("COMPARATIVE EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('comparison_results.csv', index=False)
    print("\nResults saved to comparison_results.csv")

if __name__ == "__main__":
    main()
