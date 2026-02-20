"""
Download Reddit data for Paper 4 Dashboard - Solar, Electric Vehicles, and Vegan subreddits
INCREMENTAL VERSION: Only downloads new data that doesn't already exist
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime, timedelta
import praw
import time
from tqdm import tqdm
import warnings
import hashlib
warnings.filterwarnings('ignore')

def initialize_reddit():
    """Initialize Reddit API client"""
    try:
        reddit = praw.Reddit(
            client_id="lnXXirKmeS9FsDoGDwELIg",
            client_secret="ANvb8sWAQSWDW8rNQm1N5cB5FxvVFg",
            user_agent='lowAI by u/sword-in-stone'
        )
        print("✓ Reddit API initialized successfully")
        print(f"Read-only: {reddit.read_only}")
        return reddit
    except Exception as e:
        print(f"❌ Failed to initialize Reddit API: {e}")
        return None

def load_existing_data(output_dir='paper4dashboard_data'):
    """Load all existing data to check for duplicates"""
    
    existing_data = {
        'posts': set(),
        'comments': set(),
        'text_hashes': set()  # Hash of text content to avoid exact duplicates
    }
    
    # Find all existing CSV files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv') and 'paper4_' in f]
    
    for csv_file in csv_files:
        try:
            file_path = os.path.join(output_dir, csv_file)
            df = pd.read_csv(file_path)
            
            print(f"  Loading existing data from {csv_file} ({len(df)} items)")
            
            # Add IDs to existing sets
            if 'id' in df.columns:
                for _, row in df.iterrows():
                    existing_data['posts'].add(row['id'])
                    
                    # Create hash of text content for duplicate detection
                    text_content = f"{row.get('title', '')} {row.get('text', '')}".strip()
                    if text_content:
                        text_hash = hashlib.md5(text_content.encode()).hexdigest()
                        existing_data['text_hashes'].add(text_hash)
                        
        except Exception as e:
            print(f"    ⚠️ Error loading {csv_file}: {e}")
            continue
    
    print(f"✓ Loaded {len(existing_data['posts'])} existing post IDs")
    print(f"✓ Loaded {len(existing_data['text_hashes'])} existing text hashes")
    
    return existing_data

def download_subreddit_data_incremental(reddit, subreddit_name, existing_data, limit=1000, time_filter='month'):
    """Download posts and comments from a subreddit, skipping existing ones"""
    
    print(f"📥 Downloading new data from r/{subreddit_name}...")
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        
        # Download posts
        posts_data = []
        new_posts = 0
        skipped_posts = 0
        
        print(f"  Downloading posts (limit: {limit})...")
        
        for post in tqdm(subreddit.top(time_filter=time_filter, limit=limit), total=limit):
            try:
                # Check if post already exists
                if post.id in existing_data['posts']:
                    skipped_posts += 1
                    continue
                
                # Create text hash to check for exact duplicates
                text_content = f"{post.title} {post.selftext if post.selftext else ''}".strip()
                text_hash = hashlib.md5(text_content.encode()).hexdigest()
                
                if text_hash in existing_data['text_hashes']:
                    skipped_posts += 1
                    continue
                
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext if post.selftext else '',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'author': str(post.author) if post.author else '[deleted]',
                    'subreddit': subreddit_name,
                    'content_type': 'post',
                    'url': post.url,
                    'permalink': post.permalink,
                    'text_hash': text_hash
                }
                posts_data.append(post_data)
                new_posts += 1
                
                # Add to existing data to avoid duplicates within this session
                existing_data['posts'].add(post.id)
                existing_data['text_hashes'].add(text_hash)
                
            except Exception as e:
                print(f"    ⚠️ Error processing post {post.id}: {e}")
                continue
        
        print(f"    ✓ Downloaded {new_posts} new posts, skipped {skipped_posts} existing posts")
        
        # Download comments
        comments_data = []
        new_comments = 0
        skipped_comments = 0
        
        print(f"  Downloading comments (limit: {limit})...")
        
        comment_count = 0
        for post in tqdm(subreddit.top(time_filter=time_filter, limit=limit), total=limit):
            try:
                post.comments.replace_more(limit=0)  # Remove MoreComments objects
                for comment in post.comments.list():
                    if comment_count >= limit:
                        break
                    
                    try:
                        # Check if comment already exists
                        if comment.id in existing_data['posts']:  # Using same set for simplicity
                            skipped_comments += 1
                            comment_count += 1
                            continue
                        
                        # Create text hash to check for exact duplicates
                        text_content = comment.body.strip()
                        text_hash = hashlib.md5(text_content.encode()).hexdigest()
                        
                        if text_hash in existing_data['text_hashes']:
                            skipped_comments += 1
                            comment_count += 1
                            continue
                        
                        comment_data = {
                            'id': comment.id,
                            'title': '',  # Comments don't have titles
                            'text': comment.body,
                            'score': comment.score,
                            'upvote_ratio': 1.0,  # Comments don't have upvote_ratio
                            'num_comments': 0,  # Comments don't have num_comments
                            'created_utc': comment.created_utc,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'subreddit': subreddit_name,
                            'content_type': 'comment',
                            'url': '',  # Comments don't have URLs
                            'permalink': comment.permalink,
                            'parent_id': comment.parent_id,
                            'text_hash': text_hash
                        }
                        comments_data.append(comment_data)
                        new_comments += 1
                        
                        # Add to existing data to avoid duplicates within this session
                        existing_data['posts'].add(comment.id)
                        existing_data['text_hashes'].add(text_hash)
                        
                        comment_count += 1
                    except Exception as e:
                        print(f"    ⚠️ Error processing comment {comment.id}: {e}")
                        comment_count += 1
                        continue
                        
                if comment_count >= limit:
                    break
                    
            except Exception as e:
                print(f"    ⚠️ Error processing post comments {post.id}: {e}")
                continue
        
        print(f"    ✓ Downloaded {new_comments} new comments, skipped {skipped_comments} existing comments")
        
        return posts_data, comments_data
        
    except Exception as e:
        print(f"❌ Error downloading from r/{subreddit_name}: {e}")
        return [], []

def save_data_to_files(posts_data, comments_data, output_dir='paper4dashboard_data'):
    """Save downloaded data to files with timestamp"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save posts
    if posts_data:
        posts_df = pd.DataFrame(posts_data)
        posts_csv_file = os.path.join(output_dir, f'paper4_posts_{timestamp}.csv')
        posts_json_file = os.path.join(output_dir, f'paper4_posts_{timestamp}.json')
        
        posts_df.to_csv(posts_csv_file, index=False)
        posts_df.to_json(posts_json_file, orient='records', indent=2)
        
        print(f"✓ Saved {len(posts_data)} posts to {posts_csv_file}")
    
    # Save comments
    if comments_data:
        comments_df = pd.DataFrame(comments_data)
        comments_csv_file = os.path.join(output_dir, f'paper4_comments_{timestamp}.csv')
        comments_json_file = os.path.join(output_dir, f'paper4_comments_{timestamp}.json')
        
        comments_df.to_csv(comments_csv_file, index=False)
        comments_df.to_json(comments_json_file, orient='records', indent=2)
        
        print(f"✓ Saved {len(comments_data)} comments to {comments_csv_file}")
    
    # Combine all data
    all_data = posts_data + comments_data
    if all_data:
        combined_df = pd.DataFrame(all_data)
        combined_csv_file = os.path.join(output_dir, f'paper4_combined_{timestamp}.csv')
        combined_df.to_csv(combined_csv_file, index=False)
        
        print(f"✓ Saved {len(all_data)} total items to {combined_csv_file}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_items': len(all_data),
            'posts': len(posts_data),
            'comments': len(comments_data),
            'subreddits': list(set(item['subreddit'] for item in all_data)),
            'content_types': list(set(item['content_type'] for item in all_data))
        }
        
        metadata_file = os.path.join(output_dir, f'paper4_metadata_{timestamp}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {metadata_file}")
    
    return timestamp

def main():
    """Main function to download data from target subreddits"""
    
    print("="*80)
    print("PAPER 4 DASHBOARD - INCREMENTAL DATA DOWNLOAD")
    print("="*80)
    
    # Initialize Reddit API
    reddit = initialize_reddit()
    if not reddit:
        return
    
    # Load existing data to avoid duplicates
    print("\n📋 Loading existing data...")
    existing_data = load_existing_data()
    
    # Target subreddits
    target_subreddits = ['solar', 'electricvehicles', 'vegan']
    
    all_posts_data = []
    all_comments_data = []
    
    # Download data from each subreddit
    for subreddit_name in target_subreddits:
        print(f"\n{'='*60}")
        print(f"PROCESSING: r/{subreddit_name}")
        print(f"{'='*60}")
        
        posts_data, comments_data = download_subreddit_data_incremental(
            reddit, subreddit_name, existing_data, limit=1000, time_filter='month'
        )
        
        all_posts_data.extend(posts_data)
        all_comments_data.extend(comments_data)
        
        # Add delay to be respectful to Reddit API
        time.sleep(2)
    
    # Save all data
    if all_posts_data or all_comments_data:
        print(f"\n{'='*60}")
        print("SAVING DATA")
        print(f"{'='*60}")
        
        timestamp = save_data_to_files(all_posts_data, all_comments_data)
        
        print(f"\n✓ Download complete!")
        print(f"✓ Total new posts: {len(all_posts_data)}")
        print(f"✓ Total new comments: {len(all_comments_data)}")
        print(f"✓ Timestamp: {timestamp}")
    else:
        print("\n✓ No new data to download - all data already exists!")

if __name__ == "__main__":
    main() 