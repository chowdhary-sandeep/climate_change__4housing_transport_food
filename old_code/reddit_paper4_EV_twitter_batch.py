"""
Apply trained RoBERTa models to Twitter data in batches for memory efficiency
Processes Twitter data from sector_data_fast directory in manageable chunks
"""

import os
import pandas as pd
import numpy as np
import glob
import torch
from tqdm import tqdm
from evaluate_trained_models import (
    load_trained_model, 
    load_test_data, 
    predict_with_scores
)

def load_twitter_data_batch(data_dir="paper4data/sector_data_fast", batch_size=1000):
    """Load Twitter data in batches from sector_data_fast directory"""
    
    print("Loading Twitter data from sector_data_fast in batches...")
    
    # Load all summary files (main analysis files)
    summary_files = glob.glob(os.path.join(data_dir, "summary_batch_*.parquet"))
    summary_files.sort()  # Ensure consistent ordering
    
    if not summary_files:
        print(f"No summary files found in {data_dir}")
        return None
    
    print(f"Found {len(summary_files)} summary files")
    
    # Process each batch file separately
    batch_data = []
    for file in tqdm(summary_files, desc="Loading batch files"):
        df = pd.read_parquet(file)
        batch_data.append(df)
    
    return batch_data

def process_twitter_batch_with_models(batch_data, 
                                    save_dir="paper4data/sectorwise_roberta_classifications_twitter",
                                    thresholded_save_dir="paper4data/sectorwise_roberta_classifications_thresholded_twitter",
                                    batch_size=1000):
    """Process Twitter data in batches with model predictions"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find and load all trained models
    model_dirs = glob.glob("models/top7_*_distilbert_base_uncased")
    
    if not model_dirs:
        print("No trained models found!")
        return None, None
    
    # Load all sector models
    all_sector_models = {}
    optimal_thresholds = {}
    sector_labels = {}
    
    for model_dir in model_dirs:
        # Extract sector from directory name
        sector = model_dir.split('_')[1]  # top7_SECTOR_model
        
        print(f"Loading {sector.upper()} model...")
        
        # Load model
        model, tokenizer, label_names, sector_optimal_thresholds = load_trained_model(model_dir, device)
        
        if model is None:
            print(f"Failed to load {sector} model")
            continue
            
        all_sector_models[sector] = {
            'model': model,
            'tokenizer': tokenizer,
            'label_names': label_names
        }
        
        optimal_thresholds[sector] = sector_optimal_thresholds
        sector_labels[sector] = label_names
    
    if not all_sector_models:
        print("No models loaded successfully!")
        return None, None
        
    # Ensure output directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(thresholded_save_dir, exist_ok=True)
    
    # Initialize sector-specific result collectors
    sector_results = {sector: [] for sector in sector_labels.keys()}
    sector_thresholded_results = {sector: [] for sector in sector_labels.keys()}
    
    # Process each batch
    for batch_idx, batch_df in enumerate(tqdm(batch_data, desc="Processing batches")):
        print(f"\nProcessing batch {batch_idx + 1}/{len(batch_data)} ({len(batch_df)} tweets)")
        
        # Group tweets by sector within this batch
        for sector in sector_labels.keys():
            sector_tweets = batch_df[batch_df['primary_sector'] == sector]['text'].tolist()
            
            if not sector_tweets:
                continue
                
            print(f"  Processing {len(sector_tweets)} {sector} tweets...")
            
            model = all_sector_models[sector]['model']
            tokenizer = all_sector_models[sector]['tokenizer']
            label_names = all_sector_models[sector]['label_names']
            
            # Process tweets in smaller chunks within the batch
            for i in range(0, len(sector_tweets), batch_size):
                chunk = sector_tweets[i:i + batch_size]
                
                for text in tqdm(chunk, desc=f"  {sector} chunk {i//batch_size + 1}", leave=False):
                    # Get predictions
                    prediction_results = predict_with_scores(text, model, tokenizer, label_names, device)
                    scores = [res['score'] for res in prediction_results]
                    
                    # Create result dict
                    result = {'sector': sector, 'text': text, 'batch_idx': batch_idx}
                    
                    # Add scores for each label
                    for label, score in zip(label_names, scores):
                        result[f"{sector}_{label}"] = score
                        
                    sector_results[sector].append(result)
                    
                    # Create thresholded version
                    thresholded_row = {'sector': sector, 'text': text, 'batch_idx': batch_idx}
                    thresholds = optimal_thresholds.get(sector, {})
                    for label in label_names:
                        col = f"{sector}_{label}"
                        score = result[col]
                        threshold = thresholds.get(label, 0.5)
                        thresholded_row[col] = int(score > threshold)
                    sector_thresholded_results[sector].append(thresholded_row)
    
    # Save results for each sector
    print("\n=== Saving Results ===")
    for sector in sector_labels.keys():
        if not sector_results[sector]:
            print(f"No results for {sector}, skipping...")
            continue
            
        # Save raw results
        sector_df = pd.DataFrame(sector_results[sector])
        sector_csv_path = os.path.join(save_dir, f"{sector}_Roberta_classifications_twitter.csv")
        sector_df.to_csv(sector_csv_path, index=False)
        print(f"Saved {sector} Twitter Roberta classifications to {sector_csv_path}")
        
        # Save thresholded results
        thresholded_sector_df = pd.DataFrame(sector_thresholded_results[sector])
        thresholded_sector_csv_path = os.path.join(thresholded_save_dir, f"{sector}_Roberta_classifications_thresholded_twitter.csv")
        thresholded_sector_df.to_csv(thresholded_sector_csv_path, index=False)
        print(f"Saved thresholded {sector} Twitter Roberta classifications to {thresholded_sector_csv_path}")
    
    # Combine all results
    all_results = []
    all_thresholded_results = []
    for sector in sector_labels.keys():
        all_results.extend(sector_results[sector])
        all_thresholded_results.extend(sector_thresholded_results[sector])
    
    results_df = pd.DataFrame(all_results)
    thresholded_df = pd.DataFrame(all_thresholded_results)
    
    return results_df, sector_labels

def main():
    """Main function to run Twitter analysis in batches"""
    
    print("=== Twitter Data Analysis with Trained RoBERTa Models (Batch Processing) ===")
    
    # Load Twitter data in batches
    batch_data = load_twitter_data_batch()
    
    if batch_data is None:
        print("Failed to load Twitter data. Exiting.")
        return
    
    # Process with model predictions
    print("\n=== Running Model Predictions ===")
    results_df, sector_labels = process_twitter_batch_with_models(batch_data)
    
    if results_df is not None:
        print(f"\n=== Analysis Complete ===")
        print(f"Total tweets processed: {len(results_df)}")
        print(f"Sectors processed: {list(sector_labels.keys())}")
        
        # Save combined results
        combined_save_path = "paper4data/twitter_roberta_classifications_combined_batch.csv"
        results_df.to_csv(combined_save_path, index=False)
        print(f"Combined results saved to: {combined_save_path}")
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for sector in sector_labels.keys():
            sector_data = results_df[results_df['sector'] == sector]
            print(f"{sector}: {len(sector_data)} tweets")
    else:
        print("Failed to generate predictions. Check model files and data.")

if __name__ == "__main__":
    main()
