"""
Apply trained RoBERTa models to Twitter data from sector_data_fast
Based on the Reddit analysis but adapted for Twitter data structure
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

def load_twitter_data(data_dir="paper4data/sector_data_fast"):
    """Load Twitter data from sector_data_fast directory"""
    
    print("Loading Twitter data from sector_data_fast...")
    
    # Load all summary files (main analysis files)
    summary_files = glob.glob(os.path.join(data_dir, "summary_batch_*.parquet"))
    summary_files.sort()  # Ensure consistent ordering
    
    if not summary_files:
        print(f"No summary files found in {data_dir}")
        return None, None
    
    print(f"Found {len(summary_files)} summary files")
    
    # Load and combine all summary data
    all_summaries = []
    for file in tqdm(summary_files, desc="Loading summary files"):
        df = pd.read_parquet(file)
        all_summaries.append(df)
    
    combined_df = pd.concat(all_summaries, ignore_index=True)
    print(f"Loaded {len(combined_df)} total tweets")
    
    # Group tweets by sector based on primary_sector column
    tweets_by_sector = {}
    for sector in ['transport_strong', 'transport_weak', 'housing_strong', 'housing_weak', 'food_strong', 'food_weak']:
        sector_tweets = combined_df[combined_df['primary_sector'] == sector]['text'].tolist()
        tweets_by_sector[sector] = set(sector_tweets)  # Use set to remove duplicates
        print(f"{sector}: {len(sector_tweets)} tweets")
    
    return combined_df, tweets_by_sector

def get_twitter_model_predictions(tweets_by_sector, 
                                 save_dir="paper4data/sectorwise_roberta_classifications_twitter",
                                 thresholded_save_dir="paper4data/sectorwise_roberta_classifications_thresholded_twitter"):
    """Get model predictions for Twitter data, save raw and thresholded results to disk, one CSV per sector"""
    
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
    
    # Store all results for return
    all_results = []
    all_thresholded_results = []
    
    # Process each sector
    for sector, tweets in tweets_by_sector.items():
        if sector not in all_sector_models:
            print(f"No model found for {sector}, skipping...")
            continue
            
        print(f"\nProcessing {sector} tweets ({len(tweets)} tweets)...")
        
        model = all_sector_models[sector]['model']
        tokenizer = all_sector_models[sector]['tokenizer']
        label_names = all_sector_models[sector]['label_names']
        
        # Convert set to list for processing
        tweet_list = list(tweets)
        
        # Get predictions for each tweet
        sector_results = []
        sector_thresholded_results = []
        
        for text in tqdm(tweet_list, desc=f"Processing {sector}"):
            # Get predictions
            prediction_results = predict_with_scores(text, model, tokenizer, label_names, device)
            scores = [res['score'] for res in prediction_results]
            
            # Create result dict
            result = {'sector': sector, 'text': text}
            
            # Add scores for each label
            for label, score in zip(label_names, scores):
                result[f"{sector}_{label}"] = score
                
            sector_results.append(result)
        
        # Save raw results for this sector
        sector_df = pd.DataFrame(sector_results)
        sector_csv_path = os.path.join(save_dir, f"{sector}_Roberta_classifications_twitter.csv")
        sector_df.to_csv(sector_csv_path, index=False)
        print(f"Saved {sector} Twitter Roberta classifications to {sector_csv_path}")
        all_results.extend(sector_results)
        
        # Now create thresholded version for this sector
        thresholds = optimal_thresholds.get(sector, {})
        for row in sector_results:
            thresholded_row = {'sector': sector, 'text': row['text']}
            for label in label_names:
                col = f"{sector}_{label}"
                score = row[col]
                # Use optimal threshold if available, else 0.5
                threshold = thresholds.get(label, 0.5)
                thresholded_row[col] = int(score > threshold)
            sector_thresholded_results.append(thresholded_row)
            
        thresholded_sector_df = pd.DataFrame(sector_thresholded_results)
        thresholded_sector_csv_path = os.path.join(thresholded_save_dir, f"{sector}_Roberta_classifications_thresholded_twitter.csv")
        thresholded_sector_df.to_csv(thresholded_sector_csv_path, index=False)
        print(f"Saved thresholded {sector} Twitter Roberta classifications to {thresholded_sector_csv_path}")
        all_thresholded_results.extend(sector_thresholded_results)
    
    # Optionally, return all results as a single dataframe for further use
    results_df = pd.DataFrame(all_results)
    thresholded_df = pd.DataFrame(all_thresholded_results)
    
    return results_df, sector_labels

def main():
    """Main function to run Twitter analysis"""
    
    print("=== Twitter Data Analysis with Trained RoBERTa Models ===")
    
    # Load Twitter data
    twitter_df, tweets_by_sector = load_twitter_data()
    
    if twitter_df is None:
        print("Failed to load Twitter data. Exiting.")
        return
    
    # Get model predictions
    print("\n=== Running Model Predictions ===")
    results_df, sector_labels = get_twitter_model_predictions(tweets_by_sector)
    
    if results_df is not None:
        print(f"\n=== Analysis Complete ===")
        print(f"Total tweets processed: {len(results_df)}")
        print(f"Sectors processed: {list(sector_labels.keys())}")
        
        # Save combined results
        combined_save_path = "paper4data/twitter_roberta_classifications_combined.csv"
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
