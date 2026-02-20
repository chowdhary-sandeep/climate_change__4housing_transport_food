"""
Create interactive dashboard for Paper 4 sector analysis results
Focus: r/solar, r/electricvehicles, r/vegan with sector-based classifiers
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

def load_latest_analysis_data(input_dir='paper4dashboard_data'):
    """Load the latest analysis data"""
    
    # Find the latest analysis files
    analysis_files = [f for f in os.listdir(input_dir) if f.startswith('paper4_sector_analysis_summary_')]
    if not analysis_files:
        print("❌ No analysis summary files found")
        return None, None, None, None
    
    # Get the latest file
    latest_file = sorted(analysis_files)[-1]
    print(f"✓ Latest analysis file: {latest_file}")
    
    # Extract timestamp
    timestamp = latest_file.replace('paper4_sector_analysis_summary_', '').replace('.csv', '')
    print(f"✓ Loading analysis data from timestamp: {timestamp}")
    
    # Load summary data
    summary_file = os.path.join(input_dir, latest_file)
    summary_df = pd.read_csv(summary_file)
    
    print(f"✓ Summary file columns: {list(summary_df.columns)}")
    print(f"✓ Summary file shape: {summary_df.shape}")
    print(f"✓ First few rows:")
    print(summary_df.head())
    
    # Load sample posts
    sample_posts_file = os.path.join(input_dir, f'paper4_sample_posts_{timestamp}.json')
    sample_posts = {}
    if os.path.exists(sample_posts_file):
        with open(sample_posts_file, 'r') as f:
            sample_posts = json.load(f)
    
    # Load overall stats
    stats_file = os.path.join(input_dir, f'paper4_overall_stats_{timestamp}.json')
    overall_stats = {}
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            overall_stats = json.load(f)
    
    return summary_df, sample_posts, overall_stats, timestamp

def create_hover_text(subreddit, content_type, sector, label, sample_posts):
    """Create hover text with sample posts for a specific subreddit-content_type-sector-label combination"""
    key = f"{subreddit}_{content_type}_{sector}"
    
    if key not in sample_posts or not sample_posts[key]:
        return f"r/{subreddit}<br>{content_type}<br>{sector}<br>{label}<br>No sample posts available"
    
    hover_text = f"<b>r/{subreddit}</b><br><b>{content_type.title()}</b><br><b>{sector.title()}</b><br><b>{label.title()}</b><br><br>"
    hover_text += "<b>Sample Posts:</b><br>"
    
    # Get posts that1 have high scores for this specific label
    label_posts = []
    for post in sample_posts[key]:
        if 'scores' in post and label in post['scores']:
            label_score = post['scores'][label]
            if label_score > 0.5:  # Only include posts with high scores for this label
                label_posts.append((post, label_score))
    
    # Sort by label score and take top 3
    label_posts.sort(key=lambda x: x[1], reverse=True)
    label_posts = label_posts[:3]
    
    if not label_posts:
        hover_text += "No high-scoring posts for this label"
    else:
        for i, (post, label_score) in enumerate(label_posts, 1):
            # Handle NaN values and truncate title and text to keep it concise
            title = str(post.get('title', 'No title')) if pd.notna(post.get('title')) else 'No title'
            title = title[:100] + "..." if len(title) > 100 else title
            
            text = str(post.get('text', 'No text')) if pd.notna(post.get('text')) else 'No text'
            text = text[:150] + "..." if len(text) > 150 else text
            
            hover_text += f"<br><span style='font-size: 10px;'><b>{i}.</b> {title}</span><br>"
            hover_text += f"<span style='font-size: 9px; color: #666;'>{text}</span><br>"
            hover_text += f"<span style='font-size: 9px;'><b>{label.title()} Score:</b> {label_score:.3f}</span><br>"
    
    return hover_text

def create_sector_dashboards(summary_df, sample_posts, timestamp, output_dir='paper4dashboard_data'):
    """Create combined sector dashboard with all 3 sectors in one HTML"""
    
    print("Creating combined sector dashboard...")
    
    # Define color palettes for each sector using ColorBrewer schemes
    color_palettes = {
        'transport': ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d'],  # BuPu (Blue-Purple)
        'housing': ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000'],    # OrRd (Orange-Red)
        'food': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45']        # GnBu (Green-Blue)
    }
    
    # Create subplots with 1 row and 3 columns
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Transport Sector', 'Housing Sector', 'Food Sector'],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.05
    )
    
    # Create plots for each sector
    for col, sector in enumerate(['transport', 'housing', 'food'], 1):
        # Get data for this sector
        sector_data = summary_df[summary_df['sector'] == sector].copy()
        
        if sector_data.empty:
            print(f"⚠️ No data for {sector} sector")
            continue
        
        print(f"✓ {sector} sector: Found {len(sector_data)} labels")
        
        # Create bar chart for this sector
        labels = sector_data['label'].tolist()
        positive_percentages = sector_data['positive_percentage'].tolist()
        total_items = sector_data['total_items'].tolist()
        
        # Create hover text for each label
        hover_texts = []
        for i, (label, percentage, total) in enumerate(zip(labels, positive_percentages, total_items)):
            hover_text = f"<b>{sector.title()} Sector</b><br><b>{label.title()}</b><br>"
            hover_text += f"Positive: {percentage:.1f}%<br>"
            hover_text += f"Total items: {total}<br>"
            hover_text += f"Mean score: {sector_data.iloc[i]['mean_score']:.3f}"
            hover_texts.append(hover_text)
        
        # Add bars for this sector
        fig.add_trace(
            go.Bar(
                x=labels,
                y=positive_percentages,
                name=sector.title(),
                hovertemplate=hover_texts,
                hoverinfo='text',
                marker_color=color_palettes[sector][0],  # Use first color from palette
                showlegend=True
            ),
            row=1, col=col
        )
    
    # Update overall layout
    fig.update_layout(
        title={
            'text': f'Sector Analysis Dashboard<br><sub>Analysis timestamp: {timestamp}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=500,
        width=1200,  # Make it wider to accommodate 3 plots
        barmode='stack',
        showlegend=True,
        hovermode='closest'
    )
    
    # Update axes labels for each subplot
    for i in range(1, 4):
        fig.update_xaxes(title_text="Labels", row=1, col=i)
        fig.update_yaxes(title_text="Positive Percentage (%)", row=1, col=i)
    
    # Save combined dashboard
    dashboard_file = os.path.join(output_dir, f'paper4_combined_sector_dashboard_{timestamp}.html')
    fig.write_html(dashboard_file)
    print(f"✓ Combined sector dashboard saved to: {dashboard_file}")
    
    return dashboard_file

def create_summary_table(summary_df, overall_stats, timestamp, output_dir='paper4dashboard_data'):
    """Create a summary table with sector analysis info"""
    
    print("Creating summary table...")
    
    # Create a table with key metrics for each sector and label
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Sector', 'Label', 'Total Items', 'Positive %', 'Mean Score', 'Median Score', 'Std Score'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                summary_df['sector'],
                summary_df['label'],
                summary_df['total_items'],
                [f"{p:.1f}%" for p in summary_df['positive_percentage']],
                [f"{s:.3f}" for s in summary_df['mean_score']],
                [f"{s:.3f}" for s in summary_df['median_score']],
                [f"{s:.3f}" for s in summary_df['std_score']]
            ],
            fill_color='white',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title=f'Paper 4 Sector Analysis Summary Table - {timestamp}',
        height=600
    )
    
    table_file = os.path.join(output_dir, f'paper4_sector_analysis_table_{timestamp}.html')
    fig.write_html(table_file)
    print(f"✓ Summary table saved to: {table_file}")
    
    return fig

# Helper functions removed as they're no longer needed with the new data structure

def main():
    """Main function to create the sector analysis dashboard"""
    
    print("="*80)
    print("PAPER 4 DASHBOARD - SECTOR ANALYSIS VISUALIZER")
    print("="*80)
    print("Sectors: transport, housing, food")
    print("="*80)
    
    # Load latest analysis data
    summary_df, sample_posts, overall_stats, timestamp = load_latest_analysis_data()
    
    if summary_df is None:
        print("❌ Failed to load analysis data. Please run the analysis first.")
        return
    
    # Create combined sector dashboard
    dashboard_file = create_sector_dashboards(summary_df, sample_posts, timestamp)
    
    # Create summary table
    create_summary_table(summary_df, overall_stats, timestamp)
    
    print(f"\n" + "="*80)
    print("COMBINED DASHBOARD CREATION COMPLETE!")
    print("="*80)
    print(f"Combined sector dashboard: {dashboard_file}")
    print(f"Summary table: paper4dashboard_data/paper4_sector_analysis_table_{timestamp}.html")
    print(f"Analysis timestamp: {timestamp}")
    print(f"Color palettes applied:")
    print(f"  Transport: BuPu (Blue-Purple)")
    print(f"  Housing: OrRd (Orange-Red)") 
    print(f"  Food: GnBu (Green-Blue)")
    print(f"Features:")
    print(f"  - 3 horizontal sector charts in one HTML")
    print(f"  - Distinct ColorBrewer palettes for each sector")
    print(f"  - Interactive hover text with metrics")

if __name__ == "__main__":
    main() 