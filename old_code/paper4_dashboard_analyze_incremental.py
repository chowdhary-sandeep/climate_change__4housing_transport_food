"""
Analyze Reddit data for Paper 4 Dashboard using sector-based classifiers
INCREMENTAL VERSION: Only analyzes new data that hasn't been processed yet
Focus: r/solar, r/electricvehicles, r/vegan
"""

import os
import sys
import pandas as pd
import json
import numpy as np
from datetime import datetime
import glob
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the sector-based classifier functions
try:
    from evaluate_trained_models import (
        load_trained_model, 
        load_test_data, 
        predict_with_scores
    )
except ImportError as e:
    print(f"❌ Error importing evaluate_trained_models: {e}")
    print("Please ensure evaluate_trained_models.py is in the current directory")
    sys.exit(1)

def get_sector_labels_from_models():
    """
    Finds all model directories and loads each model to extract its label names.
    Returns a dict: {sector: [label1, label2, ...], ...}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Find all trained model directories
    model_dirs = glob.glob("models/top7_*_distilbert_base_uncased")
    sector_labels = {}
    
    print("🔍 Loading sector models...")
    for model_dir in model_dirs:
        # Extract sector from directory name
        sector = model_dir.split('_')[1]  # expects format: top7_SECTOR_model
        print(f"  Loading {sector} model...")
        
        # Load model, tokenizer, label_names, thresholds
        model, tokenizer, label_names, _ = load_trained_model(model_dir, device)
        if label_names is not None:
            sector_labels[sector] = label_names
            print(f"    ✓ Loaded {sector} with labels: {label_names}")
        else:
            print(f"    ❌ Failed to load {sector} model")
    
    return sector_labels

def load_latest_data(input_dir='paper4dashboard_data'):
    """Load the latest downloaded data"""
    
    # Find the latest combined data file
    combined_files = [f for f in os.listdir(input_dir) if f.startswith('paper4_combined_') and f.endswith('.csv')]
    if not combined_files:
        print("❌ No combined data files found")
        return None, None
    
    # Get the latest file
    latest_file = sorted(combined_files)[-1]
    print(f"✓ Latest data file: {latest_file}")
    
    # Extract timestamp
    timestamp = latest_file.replace('paper4_combined_', '').replace('.csv', '')
    print(f"✓ Loading data from timestamp: {timestamp}")
    
    # Load data
    data_file = os.path.join(input_dir, latest_file)
    df = pd.read_csv(data_file)
    
    print(f"✓ Loaded {len(df)} total items")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Subreddits: {df['subreddit'].unique()}")
    print(f"✓ Content types: {df['content_type'].unique()}")
    
    return df, timestamp

def load_existing_analysis_results(input_dir='paper4dashboard_data'):
    """Load existing analysis results to identify already processed items"""
    
    # Find existing analysis files
    analysis_files = [f for f in os.listdir(input_dir) if f.startswith('paper4_sector_classifications_') and f.endswith('.csv')]
    
    if not analysis_files:
        print("✓ No existing analysis results found - will process all data")
        return set(), None
    
    # Get the latest analysis file
    latest_analysis_file = sorted(analysis_files)[-1]
    print(f"✓ Found existing analysis: {latest_analysis_file}")
    
    # Load existing results
    analysis_file = os.path.join(input_dir, latest_analysis_file)
    existing_df = pd.read_csv(analysis_file)
    
    # Extract processed IDs
    processed_ids = set(existing_df['id'].astype(str))
    print(f"✓ Found {len(processed_ids)} already processed items")
    
    return processed_ids, existing_df

def filter_new_data(df, processed_ids):
    """Filter out data that has already been analyzed"""
    
    if not processed_ids:
        print("✓ No existing processed data - will analyze all items")
        return df
    
    # Convert IDs to string for comparison
    df['id_str'] = df['id'].astype(str)
    
    # Filter for new items
    new_df = df[~df['id_str'].isin(processed_ids)].copy()
    
    print(f"✓ Total items: {len(df)}")
    print(f"✓ Already processed: {len(df) - len(new_df)}")
    print(f"✓ New items to analyze: {len(new_df)}")
    
    # Remove the temporary column
    new_df = new_df.drop('id_str', axis=1)
    
    return new_df

def get_model_predictions(df, sector_labels, save_dir="paper4dashboard_data/sector_classifications"):
    """Get predictions from all sector models for the given data"""
    
    if len(df) == 0:
        print("✓ No new data to analyze")
        return pd.DataFrame()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Create results dataframe
    results_data = []
    
    # Process each sector
    for sector, labels in sector_labels.items():
        print(f"\n🔍 Analyzing {sector} sector...")
        
        # Load model for this sector
        model_dir = f"models/top7_{sector}_distilbert_base_uncased"
        model, tokenizer, label_names, optimal_thresholds = load_trained_model(model_dir, device)
        
        if model is None:
            print(f"❌ Failed to load {sector} model, skipping...")
            continue
        
        # Get predictions for all texts
        texts = df['text'].fillna('').tolist()
        
        print(f"  Processing {len(texts)} texts...")
        
        # Process in batches to avoid memory issues
        batch_size = 32
        all_predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing {sector}"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Get predictions
                predictions = predict_with_scores(model, tokenizer, batch_texts, device)
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"    ⚠️ Error processing batch {i//batch_size}: {e}")
                # Add empty predictions for failed batch
                all_predictions.extend([[0.0] * len(labels)] * len(batch_texts))
        
        # Add results to dataframe
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx < len(all_predictions):
                pred_scores = all_predictions[idx]
                
                # Create row for this item and sector
                result_row = {
                    'id': row['id'],
                    'title': row['title'],
                    'text': row['text'],
                    'subreddit': row['subreddit'],
                    'content_type': row['content_type'],
                    'sector': sector,
                    'created_utc': row['created_utc'],
                    'author': row['author'],
                    'score': row['score'],
                    'permalink': row.get('permalink', '')
                }
                
                # Add prediction scores for each label
                for label_idx, label in enumerate(labels):
                    result_row[f'{label}_score'] = pred_scores[label_idx]
                    
                    # Apply optimal threshold if available
                    if optimal_thresholds and label in optimal_thresholds:
                        threshold = optimal_thresholds[label]
                        result_row[f'{label}_predicted'] = 1 if pred_scores[label_idx] > threshold else 0
                    else:
                        result_row[f'{label}_predicted'] = 1 if pred_scores[label_idx] > 0.5 else 0
                
                results_data.append(result_row)
    
    # Create results dataframe
    if results_data:
        results_df = pd.DataFrame(results_data)
        print(f"✓ Generated {len(results_df)} classification results")
        return results_df
    else:
        print("❌ No results generated")
        return pd.DataFrame()

def combine_with_existing_results(new_results_df, existing_df, input_dir='paper4dashboard_data'):
    """Combine new results with existing analysis results"""
    
    if existing_df is None:
        return new_results_df
    
    if len(new_results_df) == 0:
        return existing_df
    
    # Combine the dataframes
    combined_df = pd.concat([existing_df, new_results_df], ignore_index=True)
    
    # Remove duplicates based on id and sector
    combined_df = combined_df.drop_duplicates(subset=['id', 'sector'], keep='last')
    
    print(f"✓ Combined results: {len(combined_df)} total items")
    print(f"✓ Original existing: {len(existing_df)} items")
    print(f"✓ New results: {len(new_results_df)} items")
    
    return combined_df

def create_summary_statistics(results_df, sector_labels, timestamp, output_dir='paper4dashboard_data'):
    """Create summary statistics and visualizations"""
    
    if len(results_df) == 0:
        print("❌ No results to summarize")
        return
    
    print(f"\n📊 Creating summary statistics...")
    
    # Create summary statistics
    summary_stats = {}
    
    for sector, labels in sector_labels.items():
        sector_data = results_df[results_df['sector'] == sector]
        
        if len(sector_data) == 0:
            continue
        
        sector_stats = {
            'total_items': len(sector_data),
            'subreddits': sector_data['subreddit'].value_counts().to_dict(),
            'content_types': sector_data['content_type'].value_counts().to_dict(),
            'label_stats': {}
        }
        
        # Calculate statistics for each label
        for label in labels:
            score_col = f'{label}_score'
            pred_col = f'{label}_predicted'
            
            if score_col in sector_data.columns:
                scores = sector_data[score_col]
                predictions = sector_data[pred_col] if pred_col in sector_data.columns else (scores > 0.5).astype(int)
                
                sector_stats['label_stats'][label] = {
                    'mean_score': float(scores.mean()),
                    'median_score': float(scores.median()),
                    'std_score': float(scores.std()),
                    'positive_count': int(predictions.sum()),
                    'positive_percentage': float((predictions.sum() / len(predictions)) * 100),
                    'max_score': float(scores.max()),
                    'min_score': float(scores.min())
                }
        
        summary_stats[sector] = sector_stats
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f'paper4_sector_analysis_summary_{timestamp}.csv')
    
    # Create summary dataframe
    summary_rows = []
    for sector, stats in summary_stats.items():
        for label, label_stats in stats['label_stats'].items():
            row = {
                'sector': sector,
                'label': label,
                'total_items': stats['total_items'],
                'mean_score': label_stats['mean_score'],
                'median_score': label_stats['median_score'],
                'std_score': label_stats['std_score'],
                'positive_count': label_stats['positive_count'],
                'positive_percentage': label_stats['positive_percentage'],
                'max_score': label_stats['max_score'],
                'min_score': label_stats['min_score']
            }
            summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_file, index=False)
        print(f"✓ Saved summary statistics to {summary_file}")
    
    # Save overall stats
    overall_stats = {
        'timestamp': timestamp,
        'total_items': len(results_df),
        'sectors': list(sector_labels.keys()),
        'subreddits': results_df['subreddit'].unique().tolist(),
        'content_types': results_df['content_type'].unique().tolist(),
        'summary_stats': summary_stats
    }
    
    stats_file = os.path.join(output_dir, f'paper4_overall_stats_{timestamp}.json')
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2, default=str)
    
    print(f"✓ Saved overall statistics to {stats_file}")
    
    return summary_stats

def save_sample_posts(results_df, sector_labels, timestamp, output_dir='paper4dashboard_data'):
    """Save sample posts with high scores for each label"""
    
    if len(results_df) == 0:
        return
    
    print(f"\n📝 Saving sample posts...")
    
    sample_posts = {}
    
    for sector, labels in sector_labels.items():
        sector_data = results_df[results_df['sector'] == sector]
        
        if len(sector_data) == 0:
            continue
        
        sector_samples = {}
        
        for label in labels:
            score_col = f'{label}_score'
            
            if score_col in sector_data.columns:
                # Get top 5 posts with highest scores for this label
                top_posts = sector_data.nlargest(5, score_col)[['id', 'title', 'text', 'subreddit', 'content_type', score_col, 'permalink']]
                
                sector_samples[label] = top_posts.to_dict('records')
        
        sample_posts[sector] = sector_samples
    
    # Save sample posts
    sample_file = os.path.join(output_dir, f'paper4_sample_posts_{timestamp}.json')
    with open(sample_file, 'w') as f:
        json.dump(sample_posts, f, indent=2, default=str)
    
    print(f"✓ Saved sample posts to {sample_file}")

def main():
    """Main function to analyze Reddit data with sector-based classifiers"""
    
    print("="*80)
    print("PAPER 4 DASHBOARD - INCREMENTAL ANALYSIS")
    print("="*80)
    
    # Load sector labels from models
    sector_labels = get_sector_labels_from_models()
    if not sector_labels:
        print("❌ No sector models found")
        return
    
    print(f"✓ Loaded {len(sector_labels)} sector models")
    
    # Load latest data
    print(f"\n📋 Loading latest data...")
    df, timestamp = load_latest_data()
    if df is None:
        return
    
    # Load existing analysis results
    print(f"\n📋 Checking existing analysis...")
    processed_ids, existing_df = load_existing_analysis_results()
    
    # Filter for new data only
    new_df = filter_new_data(df, processed_ids)
    
    if len(new_df) == 0:
        print("✓ No new data to analyze - all data already processed!")
        return
    
    # Get model predictions for new data
    print(f"\n🤖 Running sector-based classification...")
    new_results_df = get_model_predictions(new_df, sector_labels)
    
    if len(new_results_df) == 0:
        print("❌ No classification results generated")
        return
    
    # Combine with existing results
    print(f"\n📋 Combining with existing results...")
    combined_results_df = combine_with_existing_results(new_results_df, existing_df)
    
    # Save combined results
    results_file = os.path.join('paper4dashboard_data', f'paper4_sector_classifications_{timestamp}.csv')
    combined_results_df.to_csv(results_file, index=False)
    print(f"✓ Saved combined results to {results_file}")
    
    # Create summary statistics
    summary_stats = create_summary_statistics(combined_results_df, sector_labels, timestamp)
    
    # Save sample posts
    save_sample_posts(combined_results_df, sector_labels, timestamp)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Total items analyzed: {len(combined_results_df)}")
    print(f"✓ New items processed: {len(new_results_df)}")
    print(f"✓ Timestamp: {timestamp}")

if __name__ == "__main__":
    main() 