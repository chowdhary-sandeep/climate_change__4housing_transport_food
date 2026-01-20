"""
Sample 100 random Twitter tweets per sector for manual testing
Uses the same data loading logic as plot_twitter_final_data.py load_final_twitter_results function
"""

import os
import pandas as pd
import numpy as np
import glob
import json

def load_final_twitter_results_sample():
    """Load final Twitter classification results from CSV files - same as plot_twitter_final_data.py"""
    
    print("=== Loading Final Twitter Classification Results for Sampling ===")
    
    # Define sector labels (same as plot_twitter_final_data.py)
    sector_labels = {
        'food': ['Animal Welfare', 'Environmental Impact', 'Health', 'Lab Grown And Alt Proteins',
                 'Psychology And Identity', 'Systemic Vs Individual Action', 'Taste And Convenience'],
        'housing': ['Decommissioning And Waste', 'Foreign Dependence And Trade', 'Grid Stability And Storage',
                    'Land Use', 'Local Economy', 'Subsidy And Tariff Debate', 'Utility Bills'],
        'transport': ['Alternative Modes', 'Charging Infrastructure', 'Environmental Benefit',
                      'Grid Impact And Energy Mix', 'Mineral Supply Chain', 'Policy And Mandates', 'Purchase Price']
    }
    
    # Load final results for each sector (same logic as plot_twitter_final_data.py)
    all_results = []
    save_dir = "paper4data/sectorwise_roberta_classifications_twitter"
    
    for sector in sector_labels.keys():
        csv_path = os.path.join(save_dir, f"{sector}_Roberta_classifications_twitter.csv")
        
        if os.path.exists(csv_path):
            print(f"Loading {sector} results from {csv_path}")
            sector_df = pd.read_csv(csv_path)
            
            # Add sector column if not present
            if 'sector' not in sector_df.columns:
                sector_df['sector'] = sector
            
            all_results.append(sector_df)
            print(f"  Loaded {len(sector_df)} tweets for {sector}")
        else:
            print(f"Warning: {csv_path} not found")
    
    if not all_results:
        print("No final results found!")
        return None, None
    
    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal loaded: {len(results_df)} tweets across all sectors")
    
    return results_df, sector_labels

def load_optimal_thresholds(json_path='optimal_thresholds.json', fallback_path='optimal_thresholds_jaccard.json'):
    """Load per-sector optimal thresholds; fallback to jaccard file if primary missing"""
    try:
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if os.path.exists(fallback_path):
            print(f"Warning: {json_path} not found. Using {fallback_path} instead.")
            with open(fallback_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"Warning: No thresholds file found. Defaulting to 0.5 for all labels.")
        return {}
    except Exception as e:
        print(f"Error loading thresholds: {e}. Defaulting to 0.5 for all labels.")
        return {}

def get_optimal_threshold(thresholds_map, sector, label, default_threshold=0.5):
    """Fetch threshold for given sector/label; fallback to default if missing"""
    try:
        sector_map = thresholds_map.get(sector, {})
        # Label keys in JSON are case-sensitive and expected to match exactly
        return float(sector_map.get(label, default_threshold))
    except Exception:
        return default_threshold

def sample_twitter_data_for_manual_testing(sample_size=100):
    """
    Load Twitter data and sample 100 random rows per sector for manual testing
    Uses the same data as the plotting function
    """
    print("=== Sampling Twitter Data for Manual Testing ===")
    
    # Load Twitter data using the same function as plotting
    twitter_df, sector_labels = load_final_twitter_results_sample()
    
    if twitter_df is None:
        print("Failed to load Twitter data. Exiting.")
        return
    
    print(f"Total tweets loaded: {len(twitter_df)}")
    
    # Load optimal thresholds for binarization
    optimal_thresholds = load_optimal_thresholds()
    
    # Sample 100 random rows per sector and add binary columns
    sampled_data = []
    
    for sector in sector_labels.keys():
        sector_data = twitter_df[twitter_df['sector'] == sector]
        
        if len(sector_data) > 0:
            # Sample up to sample_size rows (or all if less than sample_size)
            actual_sample_size = min(sample_size, len(sector_data))
            sector_sample = sector_data.sample(n=actual_sample_size, random_state=42)
            
            # Add binary columns using optimal thresholds
            for label in sector_labels[sector]:
                raw_colname = f"{sector}_{label}"
                binary_colname = f"{sector}_{label}_binary"
                
                if raw_colname in sector_sample.columns:
                    # Use optimal threshold for binarization
                    threshold = get_optimal_threshold(optimal_thresholds, sector, label, 0.5)
                    sector_sample[binary_colname] = (sector_sample[raw_colname] > threshold).astype(int)
                    print(f"  {sector}_{label}: threshold={threshold:.3f}, binary positives={sector_sample[binary_colname].sum()}")
            
            print(f"{sector}: {len(sector_sample)} tweets sampled (from {len(sector_data)} total)")
            sampled_data.append(sector_sample)
        else:
            print(f"{sector}: No data found")
    
    if not sampled_data:
        print("No data to sample!")
        return
    
    # Combine all sampled data
    sample_df = pd.concat(sampled_data, ignore_index=True)
    print(f"Total sampled tweets: {len(sample_df)}")
    
    # Create simplified format with 3 columns: sector, text, labels
    simplified_data = []
    
    for _, row in sample_df.iterrows():
        sector = row['sector']
        text = row['text']
        
        # Find all labels that are positive (1) for this tweet
        positive_labels = []
        for label in sector_labels[sector]:
            binary_colname = f"{sector}_{label}_binary"
            if binary_colname in row and row[binary_colname] == 1:
                positive_labels.append(label)
        
        simplified_data.append({
            'sector': sector,
            'text': text,
            'labels': ', '.join(positive_labels) if positive_labels else 'None'
        })
    
    # Create simplified dataframe
    simplified_df = pd.DataFrame(simplified_data)
    
    # Save to CSV with wide text column
    output_file = "manual_test_twitter_using_REDDIT_based_classifer.csv"
    
    # Set pandas display options for better readability
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    
    simplified_df.to_csv(output_file, index=False)
    print(f"Sampled data saved to: {output_file}")
    
    # Print summary statistics
    print("\n=== Sample Summary ===")
    sector_counts = simplified_df['sector'].value_counts()
    for sector, count in sector_counts.items():
        print(f"{sector}: {count} tweets")
    
    # Show sample of data
    print(f"\nFirst few rows:")
    print(simplified_df.head())
    
    # Show label distribution
    print(f"\nLabel distribution:")
    all_labels = []
    for labels_str in simplified_df['labels']:
        if labels_str != 'None':
            all_labels.extend([label.strip() for label in labels_str.split(',')])
    
    if all_labels:
        from collections import Counter
        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common():
            print(f"  {label}: {count} tweets")
    
    return simplified_df

def main():
    """Main function to sample Twitter data for manual testing"""
    
    print("=== Twitter Data Sampling for Manual Testing ===")
    print("Sampling 100 random tweets per sector from the same data used in plotting...")
    
    # Sample the data
    sampled_df = sample_twitter_data_for_manual_testing(sample_size=100)
    
    if sampled_df is not None:
        print("\n=== Sampling Complete ===")
        print("The sampled data has been saved for manual testing with your Reddit-based classifier.")
        print("This data comes from the same source as the plotting function in plot_twitter_final_data.py")
    else:
        print("Sampling failed.")

if __name__ == "__main__":
    main()