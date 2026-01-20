"""
Create temporal stacked area plots for Twitter data using ACTUAL finetuned models
Uses 10000 labeled samples per sector and real tweet timestamps
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from adjustText import adjust_text
import glob
from tqdm import tqdm
import torch

# Import your model functions
from evaluate_trained_models import (
    load_trained_model, 
    predict_with_scores
)

def load_twitter_data_for_models(data_dir="paper4data/sector_data_fast", sample_size=None):
    """Load Twitter data efficiently by sampling first, then getting timestamps"""
    
    print("Loading Twitter data for model predictions...")
    
    # Load all summary files first
    summary_files = glob.glob(os.path.join(data_dir, "summary_batch_*.parquet"))
    summary_files.sort()
    
    if not summary_files:
        print(f"Files not found in {data_dir}")
        return None
    
    print(f"Found {len(summary_files)} summary files")
    
    # Load and combine all summary data
    all_summary_data = []
    for file in tqdm(summary_files, desc="Loading summary files"):
        df = pd.read_parquet(file)
        all_summary_data.append(df)
    
    summary_df = pd.concat(all_summary_data, ignore_index=True)
    print(f"Total tweets loaded: {len(summary_df)}")
    
    # Process tweets per sector (sample if specified, otherwise use all)
    sampled_data = []
    sectors = ['transport_strong', 'transport_weak', 'housing_strong', 'housing_weak', 'food_strong', 'food_weak']
    
    for sector in sectors:
        sector_data = summary_df[summary_df['primary_sector'] == sector]
        if len(sector_data) > 0:
            if sample_size is not None:
                sample_size_actual = min(sample_size, len(sector_data))
                sector_sample = sector_data.sample(n=sample_size_actual, random_state=42)
                print(f"{sector}: {len(sector_sample)} tweets sampled (from {len(sector_data)} total)")
            else:
                sector_sample = sector_data
                print(f"{sector}: {len(sector_sample)} tweets (full dataset)")
            sampled_data.append(sector_sample)
        else:
            print(f"{sector}: No data found")
    
    if not sampled_data:
        print("No data to process!")
        return None
    
    sample_df = pd.concat(sampled_data, ignore_index=True)
    print(f"Total tweets to process: {len(sample_df)}")
    
    # Now get timestamps only for the sampled tweets
    print("Loading timestamps for sampled tweets...")
    tweet_ids = set(sample_df['tweet_id'].tolist())
    
    # Load text_time files and filter for our sampled tweets
    text_time_files = glob.glob(os.path.join(data_dir, "text_time_batch_*.parquet"))
    text_time_files.sort()
    
    relevant_timestamps = []
    for file in tqdm(text_time_files, desc="Loading timestamps"):
        df = pd.read_parquet(file)
        # Filter to only our sampled tweet IDs
        relevant_df = df[df['id'].isin(tweet_ids)]
        if len(relevant_df) > 0:
            relevant_timestamps.append(relevant_df)
    
    if relevant_timestamps:
        text_time_df = pd.concat(relevant_timestamps, ignore_index=True)
        # Remove duplicates based on id column
        text_time_df = text_time_df.drop_duplicates(subset=['id'])
        print(f"Found timestamps for {len(text_time_df)} unique sampled tweets")
        
        # Merge with sampled data
        merged_df = pd.merge(
            sample_df,
            text_time_df[['id', 'created_at']],
            left_on='tweet_id',
            right_on='id',
            how='left'
        )
        
        # Convert created_at to datetime
        merged_df['created_at'] = pd.to_datetime(merged_df['created_at'], errors='coerce')
        merged_df['month'] = merged_df['created_at'].dt.to_period('M')
        
        # Check temporal range
        valid_timestamps = merged_df['created_at'].dropna()
        if len(valid_timestamps) > 0:
            print(f"Temporal range: {valid_timestamps.min()} to {valid_timestamps.max()}")
            print(f"Month range: {merged_df['month'].min()} to {merged_df['month'].max()}")
        
        return merged_df
    else:
        print("No timestamps found for sampled tweets")
        return None

# Use the existing functions from evaluate_trained_models.py

def save_sector_checkpoint(sector, sector_results, label_names, checkpoint_num):
    """Save checkpoint results for a sector (overwrites previous checkpoints)"""
    save_dir = "paper4data/sectorwise_roberta_classifications_twitter_checkpoints"
    thresholded_save_dir = "paper4data/sectorwise_roberta_classifications_thresholded_twitter_checkpoints"
    
    # Ensure output directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(thresholded_save_dir, exist_ok=True)
    
    # Convert to DataFrame
    sector_df = pd.DataFrame(sector_results)
    
    # Save raw results
    sector_raw_df = sector_df[['sector', 'text'] + [f"{sector}_{label}" for label in label_names]].copy()
    sector_csv_path = os.path.join(save_dir, f"{sector}_Roberta_classifications_twitter_checkpoint_{checkpoint_num}.csv")
    sector_raw_df.to_csv(sector_csv_path, index=False)
    
    # Save thresholded results
    sector_thresholded_df = sector_df[['sector', 'text'] + [f"{sector}_{label}_binary" for label in label_names]].copy()
    # Rename binary columns to remove _binary suffix
    binary_columns = {f"{sector}_{label}_binary": f"{sector}_{label}" for label in label_names}
    sector_thresholded_df = sector_thresholded_df.rename(columns=binary_columns)
    
    thresholded_sector_csv_path = os.path.join(thresholded_save_dir, f"{sector}_Roberta_classifications_thresholded_twitter_checkpoint_{checkpoint_num}.csv")
    sector_thresholded_df.to_csv(thresholded_sector_csv_path, index=False)
    
    print(f"  Checkpoint {checkpoint_num} saved: {len(sector_results)} tweets")

def save_sector_final_results(sector, sector_results, label_names):
    """Save final results for a sector"""
    save_dir = "paper4data/sectorwise_roberta_classifications_twitter"
    thresholded_save_dir = "paper4data/sectorwise_roberta_classifications_thresholded_twitter"
    
    # Ensure output directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(thresholded_save_dir, exist_ok=True)
    
    # Convert to DataFrame
    sector_df = pd.DataFrame(sector_results)
    
    # Save raw results
    sector_raw_df = sector_df[['sector', 'text'] + [f"{sector}_{label}" for label in label_names]].copy()
    sector_csv_path = os.path.join(save_dir, f"{sector}_Roberta_classifications_twitter.csv")
    sector_raw_df.to_csv(sector_csv_path, index=False)
    print(f"  Final results saved: {sector_csv_path}")
    
    # Save thresholded results
    sector_thresholded_df = sector_df[['sector', 'text'] + [f"{sector}_{label}_binary" for label in label_names]].copy()
    # Rename binary columns to remove _binary suffix
    binary_columns = {f"{sector}_{label}_binary": f"{sector}_{label}" for label in label_names}
    sector_thresholded_df = sector_thresholded_df.rename(columns=binary_columns)
    
    thresholded_sector_csv_path = os.path.join(thresholded_save_dir, f"{sector}_Roberta_classifications_thresholded_twitter.csv")
    sector_thresholded_df.to_csv(thresholded_sector_csv_path, index=False)
    print(f"  Final thresholded results saved: {thresholded_sector_csv_path}")

def get_twitter_model_predictions(twitter_df):
    """Get model predictions for Twitter data using your finetuned models"""
    
    print("Loading your finetuned models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find and load all trained models - using the same pattern as your notebook
    model_dirs = glob.glob("models/top7_*_distilbert_base_uncased")
    
    if not model_dirs:
        print("No trained models found!")
        return None, None
    
    # Load all sector models using your existing functions
    all_sector_models = {}
    optimal_thresholds = {}
    sector_labels = {}
    
    for model_dir in model_dirs:
        # Extract sector from directory name
        sector = model_dir.split('_')[1]  # top7_SECTOR_model
        
        print(f"Loading {sector.upper()} model...")
        
        # Load model using your existing function
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
    
    print(f"Loaded models for sectors: {list(all_sector_models.keys())}")
    
    # Group Twitter data by sector
    twitter_by_sector = {}
    for sector in ['food', 'housing', 'transport']:
        # Get tweets from both strong and weak variants
        sector_tweets = twitter_df[
            (twitter_df['primary_sector'] == f"{sector}_strong") | 
            (twitter_df['primary_sector'] == f"{sector}_weak")
        ].copy()
        if len(sector_tweets) > 0:
            twitter_by_sector[sector] = sector_tweets
            print(f"{sector}: {len(sector_tweets)} tweets")
    
    # Store all results
    all_results = []
    
    # Process each sector with its corresponding model
    for sector, sector_tweets in twitter_by_sector.items():
        if sector not in all_sector_models:
            print(f"No model found for {sector}")
            continue
            
        print(f"\nProcessing {sector} tweets with {sector} model...")
        
        model = all_sector_models[sector]['model']
        tokenizer = all_sector_models[sector]['tokenizer']
        label_names = all_sector_models[sector]['label_names']
        thresholds = optimal_thresholds.get(sector, {})
        
        # Initialize sector results
        sector_results = []
        total_tweets = len(sector_tweets)
        checkpoint_interval = max(1, total_tweets // 5)  # 5 checkpoints per sector
        
        # Get predictions for each tweet using your existing function
        for idx, row in tqdm(sector_tweets.iterrows(), total=len(sector_tweets), desc=f"Predicting {sector}"):
            text = row['text']
            
            # Get predictions using your existing function
            prediction_results = predict_with_scores(text, model, tokenizer, label_names, device)
            scores = [res['score'] for res in prediction_results]
            
            # Create result dict with all original data
            result = {
                'tweet_id': row['tweet_id'],
                'text': text,
                'primary_sector': row['primary_sector'],
                'matched_keywords': row['matched_keywords'],
                'keyword_count': row['keyword_count'],
                'created_at': row['created_at'],
                'month': row['month'],
                'sector': sector
            }
            
            # Add raw scores and thresholded predictions for each label
            for label, score in zip(label_names, scores):
                result[f"{sector}_{label}"] = score
                # Use optimal threshold if available, else 0.5
                threshold = thresholds.get(label, 0.5)
                result[f"{sector}_{label}_binary"] = int(score > threshold)
                
            sector_results.append(result)
            all_results.append(result)
            
            # Save checkpoint every checkpoint_interval tweets
            if len(sector_results) % checkpoint_interval == 0:
                checkpoint_num = len(sector_results) // checkpoint_interval
                save_sector_checkpoint(sector, sector_results, sector_labels[sector], checkpoint_num)
        
        # Save final sector results
        save_sector_final_results(sector, sector_results, sector_labels[sector])
        print(f"Completed {sector}: {len(sector_results)} tweets processed")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    print(f"\nTotal predictions generated: {len(results_df)}")
    
    return results_df, sector_labels

def get_sequential_palette_for_sector(sector, n_labels):
    """Get color palette for sector"""
    sector_cmaps = {
        'transport': 'Reds',
        'housing': 'Oranges',
        'food': 'Blues'
    }
    cmap_name = sector_cmaps.get(sector, 'Greys')
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(x) for x in np.linspace(0.25, 1, n_labels)]
    return colors

def plot_twitter_temporal_results(merged_df, sector_labels, time_col='month', save_dir='paper4figs'):
    """
    Plot temporal trends of label counts for Twitter data as stacked area charts
    Using REAL timestamps from Twitter data with uniformly distributed labels and inset
    """
    
    # Font size settings
    BASE_FONT_SIZE = 15
    SMALLER_FONT_SIZE = BASE_FONT_SIZE * 0.7
    
    os.makedirs(save_dir, exist_ok=True)
    n_sectors = len(sector_labels)
    fig, axes = plt.subplots(n_sectors, 1, figsize=(12, 14), dpi=300, sharex=True, sharey=False)
    if n_sectors == 1:
        axes = [axes]
    elif n_sectors == 0:
        print("No sectors to plot.")
        return
    
    # Find the global min and max month across all sectors
    all_months = merged_df['month'].dropna().sort_values().unique()
    if len(all_months) == 0:
        print("No months found in merged_df.")
        return
    
    min_month = all_months[0]
    max_month = all_months[-1]
    
    # Use the actual data range
    min_month_for_range = min_month
    max_month_for_range = max_month
    
    # Create a full range of months
    all_months_range = pd.period_range(start=min_month_for_range, end=max_month_for_range, freq='M')
    months_str = [str(m) for m in all_months_range]
    
    print(f"Twitter data temporal range: {min_month_for_range} to {max_month_for_range}")
    print(f"Total months: {len(all_months_range)}")
    
    for idx, (ax, (sector, labels)) in enumerate(zip(axes, sector_labels.items())):
        sector_df = merged_df[merged_df['sector'] == sector].copy()
        if sector_df.empty:
            print(f"No data for sector: {sector}")
            continue
        
        # Compute total volume for each label (for ordering) - use binary predictions
        label_total_counts = {}
        for label in labels:
            colname = f"{sector}_{label}_binary"
            if colname in sector_df.columns:
                label_total_counts[label] = sector_df[colname].sum()
            else:
                label_total_counts[label] = 0
        
        # Order labels by total volume (ascending: smallest at bottom, largest at top)
        ordered_labels = sorted(labels, key=lambda l: label_total_counts[l])
        
        # Compute rolling monthly counts for each label - use binary predictions
        label_time_counts = {}
        for label in ordered_labels:
            colname = f"{sector}_{label}_binary"
            if colname in sector_df.columns:
                # Group by month and sum, then rolling average (window=3, center=True)
                monthly_counts = sector_df.groupby(time_col)[colname].sum().reindex(all_months_range, fill_value=0)
                rolling_counts = monthly_counts.rolling(window=3, center=True, min_periods=1).mean()
                label_time_counts[label] = rolling_counts
            else:
                label_time_counts[label] = pd.Series(dtype=float, index=all_months_range)
        
        # Create count matrix
        count_matrix = []
        for label in ordered_labels:
            counts = label_time_counts[label].reindex(all_months_range, fill_value=0)
            count_matrix.append(counts.values)
        count_matrix = np.vstack(count_matrix)
        
        n_labels = len(ordered_labels)
        colors = get_sequential_palette_for_sector(sector, n_labels)
        
        # Get total number of comments for this sector
        sector_total_comments = len(sector_df)
        
        # Create stacked area plot
        polys = ax.stackplot(
            np.arange(len(months_str)), count_matrix, 
            labels=[label.lower() for label in ordered_labels], 
            colors=colors, 
            alpha=1.0,
            edgecolor='white', 
            linewidth=0.7
        )
        
        ax.set_facecolor('white')
        ax.grid(axis='y', color='#e5e5e5', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
        
        # Set title
        sector_intervention = {
            'transport': 'EVs',
            'housing': 'Solar',
            'food': 'Vegetarianism'
        }
        intervention = sector_intervention.get(sector, "")
        if intervention:
            sector_title = f"{sector.capitalize()} ({intervention}) - Twitter"
        else:
            sector_title = f"{sector.capitalize()} - Twitter"
        
        ax.set_title(
            sector_title,
            fontsize=BASE_FONT_SIZE,
            pad=18,
            loc='left',
            x=0.01,
            y=.8,
            fontweight='bold',
            color='#222222',
            bbox=dict(facecolor='white', edgecolor='#bbbbbb', boxstyle='round,pad=0.18', alpha=0.95)
        )
        
        # Set axis labels
        if idx == n_sectors - 1:
            ax.set_xlabel("year", fontsize=BASE_FONT_SIZE, labelpad=10, fontweight='semibold')
        ax.set_ylabel("monthly counts (rolling)", fontsize=BASE_FONT_SIZE, labelpad=10, fontweight='semibold')
        ax.tick_params(axis='x', labelsize=BASE_FONT_SIZE, rotation=0, length=0)
        ax.tick_params(axis='y', labelsize=BASE_FONT_SIZE, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Annotate labels uniformly distributed across y-axis
        annotation_texts = []
        
        # Get y-axis limits
        y_max = np.nanmax(np.sum(count_matrix, axis=0))
        y_min = 0
        y_range = y_max - y_min
        
        # Calculate cumulative counts for each label
        label_cumulative_counts = {}
        for i, label in enumerate(ordered_labels):
            colname = f"{sector}_{label}_binary"
            if colname in sector_df.columns:
                orig_monthly_counts = sector_df.groupby(time_col)[colname].sum().reindex(all_months_range, fill_value=0)
                cumulative_count = int(np.nansum(orig_monthly_counts.values))
            else:
                cumulative_count = 0
            label_cumulative_counts[label] = cumulative_count
        
        # Place labels uniformly across y-axis, but reverse the order to match stack
        # Bottom label (lowest in stack) gets highest y position, top label gets lowest
        n_labels = len(ordered_labels)
        if n_labels > 0:
            # Create uniform y positions from top to bottom, but reverse the label order
            y_positions = np.linspace(y_max * 0.9, y_min + y_range * 0.1, n_labels)
            # Reverse the label order so it matches the visual stack (bottom to top)
            ordered_labels_reversed = ordered_labels[::-1]
            
            # Find the rightmost x position (end of data)
            last_data_idx = len(months_str) - 1
            x_position = last_data_idx + 2  # Place labels to the right of the plot
            
            for i, label in enumerate(ordered_labels_reversed):
                cumulative_count = label_cumulative_counts[label]
                
                if sector_total_comments > 0:
                    pct = 100.0 * cumulative_count / sector_total_comments
                else:
                    pct = 0.0
                
                # Get the color for this label (need to find original index)
                original_index = ordered_labels.index(label)
                color = colors[original_index]
                
                annotation_str = f"{label.replace('_', ' ').capitalize()}\n n={cumulative_count:,} ({pct:.1f}%)"
                annotation = ax.text(
                    x_position, y_positions[i],
                    annotation_str,
                    va='center', ha='left',
                    fontsize=BASE_FONT_SIZE * 0.7,
                    color=color,
                    fontweight='bold',
                    alpha=1.0,
                    bbox=dict(facecolor='white', edgecolor='white', alpha=0.7, pad=0.0)
                )
                annotation_texts.append(annotation)
        
        # Set x-axis ticks (show years)
        months_years = [int(str(m)[:4]) for m in all_months_range]
        unique_years = sorted(set(months_years))
        if len(unique_years) > 10:
            step = 3
        else:
            step = 2
        year_ticks = unique_years[::step]
        
        ax.set_xticks(np.arange(len(months_str)))
        xtick_labels = []
        used_years = set()
        for i, m in enumerate(all_months_range):
            year = int(str(m)[:4])
            if year in year_ticks and months_years[i] == year and year not in used_years:
                xtick_labels.append(str(year))
                used_years.add(year)
            else:
                xtick_labels.append("")
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(axis='x', labelsize=BASE_FONT_SIZE, rotation=0, length=0)
        
        # Set individual y-axis max for each plot
        y_max = np.nanmax(np.sum(count_matrix, axis=0))
        ax.set_ylim(0, y_max * 1.08 if y_max > 0 else 1)
        
        # ----------- INSET: Fractional (percentage) stacked area plot -----------
        # For each time point, sum all label counts (can be > number of comments due to multi-labels)
        # Then, for each label, compute fraction of total label counts at that time point
        # Use same color order as main plot, no legend, no axis labels, y-axis 0-1.2 (can sum >1)
        # Place inset on left side of main axis

        # --- Compute cumulative sum of label counts up to each month ---
        # For each label, compute cumulative sum over time
        cumulative_label_counts = np.cumsum(count_matrix, axis=1)  # shape: (n_labels, n_months)
        # For all labels, compute total cumulative sum at each time point
        cumulative_total_counts = np.sum(cumulative_label_counts, axis=0)  # shape: (n_months,)

        # To avoid division by zero, set cumulative_total_counts to 1 where zero (temporarily)
        cumulative_total_counts_safe = np.where(cumulative_total_counts == 0, 1, cumulative_total_counts)
        # For each label, compute fraction of cumulative total at each time point
        cumulative_fraction_matrix = cumulative_label_counts / cumulative_total_counts_safe  # shape: (n_labels, n_months)
        # Set columns (months) where cumulative_total_counts == 0 to all zeros (no data)
        zero_cum_months = (cumulative_total_counts == 0)
        cumulative_fraction_matrix[:, zero_cum_months] = 0.0

        # --- Smooth the cumulative fraction matrix with the same rolling window (3, center=True) ---
        smoothed_cumulative_fraction_matrix = np.zeros_like(cumulative_fraction_matrix)
        for i in range(cumulative_fraction_matrix.shape[0]):
            s = pd.Series(cumulative_fraction_matrix[i])
            smoothed_cumulative_fraction_matrix[i] = s.rolling(window=3, center=True, min_periods=1).mean().values

        # Renormalize so that at each time point, sum of all label fractions is 1 (or 0 if no data)
        col_sums_cum = smoothed_cumulative_fraction_matrix.sum(axis=0)
        col_sums_cum_safe = np.where(col_sums_cum == 0, 1, col_sums_cum)
        fraction_matrix_cum = smoothed_cumulative_fraction_matrix / col_sums_cum_safe
        # Set columns where col_sums_cum==0 to all zeros (no data)
        fraction_matrix_cum[:, col_sums_cum == 0] = 0.0

        # --- Inset axes: place on left, occupying about 32% width, 40% height, left-aligned ---
        inset_width = 0.32
        inset_height = 0.40
        inset_left = 0.1
        inset_bottom = 0.21
        ax_inset = ax.inset_axes([inset_left, inset_bottom, inset_width, inset_height])

        # Plot stacked area of cumulative fractions
        ax_inset.stackplot(
            np.arange(len(months_str)), fraction_matrix_cum,
            colors=colors,
            alpha=1.0,
            edgecolor='white',
            linewidth=0.7
        )
        ax_inset.set_ylim(0, 1)
        ax_inset.set_xlim(0, len(months_str)-1)
        ax_inset.set_facecolor('white')
        
        # Add x ticks for key years (adjust based on actual data range)
        years_in_data = [int(str(m)[:4]) for m in all_months_range]
        unique_years = sorted(set(years_in_data))
        if len(unique_years) >= 4:
            # Use 4 evenly spaced years
            step = len(unique_years) // 3
            xtick_years = unique_years[::step][:4]
        else:
            # Use all available years
            xtick_years = unique_years
        
        xtick_indices = []
        xtick_labels = []
        for year in xtick_years:
            idx = next((i for i, m in enumerate(months_str) if m.startswith(str(year))), None)
            if idx is not None:
                xtick_indices.append(idx)
                xtick_labels.append(str(year))
        ax_inset.set_xticks(xtick_indices)
        ax_inset.set_xticklabels(xtick_labels, fontsize=BASE_FONT_SIZE*0.7, color='#888888')
        
        # Add y ticks at 0, 25, 50, 75, 100 percent
        ytick_vals = [0, 0.25, 0.5, 0.75, 1.0]
        ytick_labels = ['0', '25', '50', '75', '100']
        ax_inset.set_yticks(ytick_vals)
        ax_inset.set_yticklabels(ytick_labels, fontsize=BASE_FONT_SIZE*0.7, color='#888888')
        
        for spine in ax_inset.spines.values():
            spine.set_visible(False)
        for spine in ['left', 'bottom', 'top', 'right']:
            ax_inset.spines[spine].set_visible(True)
            ax_inset.spines[spine].set_color('#bbbbbb')
            ax_inset.spines[spine].set_linewidth(1.0)
        ax_inset.text(0.6, .99, "Cumulative %", fontsize=BASE_FONT_SIZE*0.85, color='k', ha='left', va='bottom', transform=ax_inset.transAxes, alpha=1)
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(os.path.join(save_dir, f"twitter_temporal_allsectors_stackedarea_vertical_FINETUNED_MODELS.pdf"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(save_dir, f"twitter_temporal_allsectors_stackedarea_vertical_FINETUNED_MODELS.png"), bbox_inches='tight', dpi=300)
    plt.close()
    print("Plotted and saved Twitter temporal trend for all sectors using FINETUNED MODELS with uniformly distributed labels and inset (3-month rolling average).")

def main():
    """Main function to create Twitter temporal plots using your finetuned models"""
    
    print("=== Twitter Temporal Analysis with YOUR FINETUNED MODELS ===")
    
    # Load Twitter data (full dataset)
    twitter_df = load_twitter_data_for_models(sample_size=None)
    
    if twitter_df is None:
        print("Failed to load Twitter data. Exiting.")
        return
    
    # Get predictions using your finetuned models
    print("\n=== Getting Predictions from Your Finetuned Models ===")
    results_df, sector_labels = get_twitter_model_predictions(twitter_df)
    
    if results_df is None:
        print("Failed to get model predictions. Exiting.")
        return
    
    # Create the plot
    print("\n=== Creating Temporal Plot with Real Timestamps and Inset ===")
    plot_twitter_temporal_results(results_df, sector_labels)
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to paper4figs/")

if __name__ == "__main__":
    main()
