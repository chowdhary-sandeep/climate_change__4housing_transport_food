"""
Pro-Anti Comment Classifier
Classifies Reddit comments into pro, anti, pro and anti, or neither categories for specific topics
"""

import re
import pickle
import json
import time
import random
import csv
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures
import threading
import os
from collections import defaultdict

# API Configuration
api_key = "sk-proj-4X1CldONLWYUuze7sCoZT3BlbkFJciLeUe9BfYhvNv1NlW4x"
client = OpenAI(api_key=api_key)

# Topic-specific pro-anti hypotheses
# ----------------------------  TRANSPORT (Electric Vehicles)  ---------------------------- #
transport_pro_anti_hypotheses = {
    "pro": (
        "Comments that mention benefits, advantages, or positive aspects of electric vehicles, OR express desire/wanting to own/buy one. "
        "Cues: 'love my Tesla', 'great choice', 'benefits', 'advantages', 'recommend', 'better than gas', "
        "'environmentally friendly', 'cost savings', 'fun to drive', 'instant torque', 'zero emissions', "
        "'future of transportation', 'clean energy', 'sustainable', 'innovative technology', 'want to buy', "
        "'would love to own', 'planning to get', 'dream car', 'saving up for', 'can't wait to buy', "
        "'benefits include', 'advantages are', 'good for', 'helpful', 'useful', 'convenient'."
    ),
    "anti": (
        "Comments that mention problems, disadvantages, or negative aspects of electric vehicles, OR express dislike/not wanting to own/buy one. "
        "Cues: 'hate EVs', 'terrible choice', 'problems', 'disadvantages', 'not worth it', 'worse than gas', "
        "'expensive', 'range anxiety', 'charging problems', 'battery issues', 'fire risk', 'grid strain', "
        "'mining damage', 'not ready', 'overhyped', 'government forcing', 'unreliable', 'maintenance nightmare', "
        "'don't want to buy', 'would never own', 'not interested in', 'avoid', 'stay away from', "
        "'problems include', 'disadvantages are', 'bad for', 'harmful', 'inconvenient', 'annoying'."
    )
}

# ----------------------------  HOUSING (Solar Power)  ---------------------------- #
housing_pro_anti_hypotheses = {
    "pro": (
        "Comments that mention benefits, advantages, or positive aspects of solar power, OR express desire/wanting to install/own solar. "
        "Cues: 'love solar', 'great investment', 'benefits', 'advantages', 'recommend', 'clean energy', "
        "'renewable', 'cost savings', 'energy independence', 'environmentally friendly', 'reduces bills', "
        "'future of energy', 'sustainable', 'green technology', 'rooftop solar', 'solar panels', 'net metering', "
        "'want to install', 'planning to get', 'saving up for panels', 'dream of solar', 'can't wait to install', "
        "'benefits include', 'advantages are', 'good for', 'helpful', 'useful', 'convenient', 'worth it'."
    ),
    "anti": (
        "Comments that mention problems, disadvantages, or negative aspects of solar power, OR express dislike/not wanting to install/own solar. "
        "Cues: 'hate solar', 'terrible investment', 'problems', 'disadvantages', 'not worth it', 'expensive', "
        "'eyesore', 'ugly', 'ruins view', 'land use', 'wildlife harm', 'intermittent', 'unreliable', "
        "'grid problems', 'subsidy waste', 'Chinese panels', 'recycling issues', 'decommissioning costs', "
        "'not efficient', 'weather dependent', 'storage problems', 'don't want to install', 'would never get', "
        "'not interested in solar', 'avoid solar', 'stay away from panels', 'problems include', 'disadvantages are', "
        "'bad for', 'harmful', 'inconvenient', 'annoying', 'not worth the money'."
    )
}

# ----------------------------  FOOD (Vegetarianism/Veganism)  ---------------------------- #
food_pro_anti_hypotheses = {
    "pro": (
        "Comments that mention benefits, advantages, or positive aspects of vegetarianism/veganism, OR express desire/wanting to try/adopt plant-based diet. "
        "Cues: 'love being vegan', 'great choice', 'benefits', 'advantages', 'recommend', 'healthier', "
        "'environmentally friendly', 'animal welfare', 'ethical', 'sustainable', 'plant-based', 'cruelty-free', "
        "'better for planet', 'reduces suffering', 'health benefits', 'energy efficient', 'water saving', "
        "'carbon footprint', 'compassionate', 'moral choice', 'want to try', 'planning to go vegan', "
        "'saving up for plant-based', 'dream of being vegan', 'can't wait to switch', 'benefits include', "
        "'advantages are', 'good for', 'helpful', 'useful', 'worth it', 'interested in'."
    ),
    "anti": (
        "Comments that mention problems, disadvantages, or negative aspects of vegetarianism/veganism, OR express dislike/not wanting to try/adopt plant-based diet. "
        "Cues: 'hate vegan food', 'terrible choice', 'problems', 'disadvantages', 'not healthy', 'expensive', "
        "'tastes bad', 'bland', 'protein deficiency', 'B12 problems', 'unnatural', 'extreme', 'cult-like', "
        "'forcing beliefs', 'not sustainable', 'soy problems', 'processed food', 'supplements needed', "
        "'cultural disrespect', 'privileged choice', 'not practical', 'don't want to try', 'would never go vegan', "
        "'not interested in plant-based', 'avoid vegan food', 'stay away from', 'problems include', 'disadvantages are', "
        "'bad for', 'harmful', 'inconvenient', 'annoying', 'not worth it', 'not for me'."
    )
}

# Dictionary mapping sectors to topics and their hypotheses
SECTOR_TOPIC_MAPPING = {
    'transport': {
        'topic': 'Electric Vehicles',
        'hypotheses': transport_pro_anti_hypotheses
    },
    'housing': {
        'topic': 'Solar Power', 
        'hypotheses': housing_pro_anti_hypotheses
    },
    'food': {
        'topic': 'Vegetarianism/Veganism',
        'hypotheses': food_pro_anti_hypotheses
    }
}

def create_pro_anti_classification_prompt(hypotheses, topic):
    """Create the pro-anti classification prompt for GPT"""
    
    prompt = f"""
You are an expert annotator. Classify Reddit comments about **{topic}** into pro, anti, pro and anti, or neither categories.

————————————————————
Classification Categories
————————————————————
1. **PRO**: Comments that mention benefits, advantages, or positive aspects of {topic}, OR express desire/wanting to own/buy/try {topic}
2. **ANTI**: Comments that mention problems, disadvantages, or negative aspects of {topic}, OR express dislike/not wanting to own/buy/try {topic}  
3. **PRO AND ANTI**: Comments that express BOTH benefits AND problems about {topic} (mixed feelings, balanced discussion, or wanting something but facing barriers)
4. **NEITHER**: Comments that are neutral, factual without opinion, or not about {topic}

————————————————————
Detailed Guidelines
————————————————————

**PRO Examples:**
{hypotheses['pro']}

**ANTI Examples:**
{hypotheses['anti']}

**PRO AND ANTI Examples:**
- "EVs are great for the environment but too expensive for most people"
- "The only thing keeping me from buying an electric vehicle is finances"
- "I'd love to get solar panels but they're too expensive for my budget"
- "Solar panels save money but they're ugly and take up space"
- "Veganism is ethical but the food is expensive and hard to find"
- "I want to go vegan but the food is too expensive and hard to find"
- "EVs are the future but the charging infrastructure isn't ready yet"
- "Solar power is great but the installation process is a nightmare"

**NEITHER Examples:**
- "What's the weather like today?"
- "I went to the store and bought groceries"
- "The new iPhone was released yesterday"
- Factual statements without opinion: "Tesla sold 500,000 cars in 2023"

————————————————————
Input format
————————————————————
Comments arrive in one block, each delimited as
[new_comment_id: XXXXXXX]
<comment text>

————————————————————
Labelling rules
————————————————————
1. Each comment must be classified into exactly ONE category: PRO, ANTI, PRO AND ANTI, or NEITHER
2. Look for explicit opinion words and sentiment indicators
3. Consider the overall tone and stance of the comment
4. If a comment mentions {topic} but is neutral/factual, classify as NEITHER
5. If a comment has mixed feelings about {topic}, classify as PRO AND ANTI
6. Ignore sarcasm unless it's clearly indicating a stance
7. Focus on the main topic - if {topic} is just mentioned in passing, consider NEITHER

————————————————————
Output format (exactly):
PRO: [comma-separated comment IDs]
ANTI: [comma-separated comment IDs]  
PRO AND ANTI: [comma-separated comment IDs]
NEITHER: [comma-separated comment IDs]

• List only IDs, no extra text.
• Omit empty brackets if no IDs match a category.
• Each comment ID should appear in exactly ONE category.

Be precise—include a comment ID only when the classification is clear.
"""
    
    return prompt

def parse_pro_anti_response(response):
    """Parse GPT response and extract comment IDs for each pro-anti category"""
    if not response:
        return {'PRO': [], 'ANTI': [], 'PRO AND ANTI': [], 'NEITHER': []}
    
    results = {'PRO': [], 'ANTI': [], 'PRO AND ANTI': [], 'NEITHER': []}
    
    # Split response into lines
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            category, ids_part = line.split(':', 1)
            category = category.strip()
            
            if category in results:
                # Extract comment IDs from brackets
                ids_text = ids_part.strip()
                if ids_text.startswith('[') and ids_text.endswith(']'):
                    ids_content = ids_text[1:-1].strip()
                    if ids_content:
                        # Split by comma and clean up
                        comment_ids = [id.strip() for id in ids_content.split(',') if id.strip()]
                        results[category] = comment_ids
    
    return results

def classify_batch_pro_anti_single_run(comments_batch, hypotheses, topic, batch_id, run_id):
    """Classify a batch of comments into pro-anti categories in a single run"""
    try:
        # Create mapping from comment to its ID within this batch
        body_to_id = {comment: f'comment{i+1}' for i, comment in enumerate(comments_batch)}
        
        # Create batch string with comments
        batch_comments = ''
        for comment in comments_batch:
            comment_id = body_to_id[comment]
            comment_str = f'\n[new_comment_id: {comment_id}]: {comment}'
            batch_comments += comment_str
        
        # Create prompt
        prompt = create_pro_anti_classification_prompt(hypotheses, topic)
        full_prompt = prompt + "\n\nComments to classify:\n" + batch_comments
        
        # Call GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert annotator for Reddit comment pro-anti classification."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        parsed_results = parse_pro_anti_response(result)
        
        # Convert comment IDs back to actual comments
        final_results = {}
        for category, comment_ids in parsed_results.items():
            matched_comments = []
            for comment_id in comment_ids:
                # Find the comment by ID
                for comment in comments_batch:
                    if body_to_id[comment] == comment_id:
                        matched_comments.append(comment)
                        break
            final_results[category] = matched_comments
        
        return {
            'topic': topic,
            'batch_id': batch_id,
            'run_id': run_id,
            'comments': comments_batch,
            'body_to_id': body_to_id,
            'raw_response': result,
            'parsed_results': final_results,
            'categories': ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']
        }
        
    except Exception as e:
        print(f"Error in batch {batch_id}, run {run_id}: {str(e)}")
        return None

def classify_batch_pro_anti_multiple_runs(comments_batch, hypotheses, topic, batch_id, num_runs=1):
    """Classify a batch of comments multiple times and calculate mean scores"""
    print(f"Processing {topic} batch {batch_id} with {len(comments_batch)} comments ({num_runs} runs)")
    
    all_runs = []
    
    for run_id in range(1, num_runs + 1):
        result = classify_batch_pro_anti_single_run(comments_batch, hypotheses, topic, batch_id, run_id)
        if result:
            all_runs.append(result)
        
        # Small delay between runs
        time.sleep(0.5)
    
    # Calculate mean scores for each comment and category
    comment_scores = defaultdict(lambda: defaultdict(list))
    
    for run_result in all_runs:
        for category, matched_comments in run_result['parsed_results'].items():
            for comment in comments_batch:
                # 1 if comment was matched to this category, 0 if not
                score = 1 if comment in matched_comments else 0
                comment_scores[comment][category].append(score)
    
    # Calculate mean scores (0/10 to 10/10)
    mean_scores = {}
    for comment in comments_batch:
        comment_mean_scores = {}
        for category in ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']:
            scores = comment_scores[comment][category]
            if scores:
                mean_score = sum(scores) / len(scores)  # This will be 0.0 to 1.0
                # Convert to 0/10 to 10/10 scale
                mean_score_10 = mean_score * 10
                comment_mean_scores[category] = mean_score_10
            else:
                comment_mean_scores[category] = 0.0
        mean_scores[comment] = comment_mean_scores
    
    return {
        'topic': topic,
        'batch_id': batch_id,
        'num_runs': num_runs,
        'all_runs': all_runs,
        'mean_scores': mean_scores,
        'categories': ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']
    }

def process_sector_comments_pro_anti(comments, hypotheses, topic, num_runs=1, num_agents=20, batch_size=10):
    """Process all comments for pro-anti classification using parallel multiple runs"""
    
    # Sample comments if too many
    sampled_comments = list(comments)[:100] if len(comments) > 100 else list(comments)
    
    if len(sampled_comments) == 0:
        return []
    
    print(f"Processing {len(sampled_comments)} comments for {topic} with {num_agents} parallel agents, batch size {batch_size}, {num_runs} runs per batch")
    
    # Split comments into batches
    comment_batches = [sampled_comments[i:i + batch_size] for i in range(0, len(sampled_comments), batch_size)]
    
    print(f"Created {len(comment_batches)} batches of {batch_size} comments each")
    
    # Create all individual batch-run tasks for full parallelization
    batch_run_tasks = []
    for batch_id, batch in enumerate(comment_batches):
        for run_id in range(1, num_runs + 1):
            batch_run_tasks.append({
                'batch_id': batch_id + 1,
                'run_id': run_id,
                'comments': batch,
                'hypotheses': hypotheses,
                'topic': topic
            })
    
    # Process all batch-run tasks in parallel
    all_run_results = []
    total_tasks = len(batch_run_tasks)
    completed_tasks = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
        # Submit all individual batch-run tasks
        future_to_task = {}
        for task in batch_run_tasks:
            future = executor.submit(
                classify_batch_pro_anti_single_run,
                task['comments'],
                task['hypotheses'],
                task['topic'],
                task['batch_id'],
                task['run_id']
            )
            future_to_task[future] = task
        
        # Use a single progress bar for all tasks
        with tqdm(total=total_tasks, desc=f"Processing {topic} batches", unit="task") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        all_run_results.append(result)
                        completed_tasks += 1
                    pbar.update(1)
                except Exception as e:
                    print(f"Error in batch {task['batch_id']} run {task['run_id']}: {str(e)}")
                    pbar.update(1)
    
    print(f"\nCompleted {completed_tasks}/{total_tasks} tasks")
    
    # Group results by batch_id and calculate mean scores
    batch_results = []
    results_by_batch = defaultdict(list)
    
    # Group all run results by batch_id
    for result in all_run_results:
        results_by_batch[result['batch_id']].append(result)
    
    # Calculate mean scores for each batch
    for batch_id, batch_runs in results_by_batch.items():
        if not batch_runs:
            continue
            
        # Get the comments for this batch from the first run
        comments_batch = batch_runs[0]['comments']
        
        # Calculate mean scores for each comment and category
        comment_scores = defaultdict(lambda: defaultdict(list))
        
        for run_result in batch_runs:
            for category, matched_comments in run_result['parsed_results'].items():
                for comment in comments_batch:
                    # 1 if comment was matched, 0 if not
                    score = 1 if comment in matched_comments else 0
                    comment_scores[comment][category].append(score)
        
        # Calculate mean scores (0/10 to 10/10)
        mean_scores = {}
        for comment in comments_batch:
            comment_mean_scores = {}
            for category in ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']:
                scores = comment_scores[comment][category]
                if scores:
                    mean_score = sum(scores) / len(scores)  # This will be 0.0 to 1.0
                    # Convert to 0/10 to 10/10 scale
                    mean_score_10 = mean_score * 10
                    comment_mean_scores[category] = mean_score_10
                else:
                    comment_mean_scores[category] = 0.0
            mean_scores[comment] = comment_mean_scores
        
        # Create consolidated batch result
        batch_result = {
            'topic': topic,
            'batch_id': batch_id,
            'num_runs': len(batch_runs),
            'all_runs': batch_runs,
            'mean_scores': mean_scores,
            'categories': ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER'],
            'comments': comments_batch
        }
        
        batch_results.append(batch_result)
    
    print(f"Processed {len(batch_results)} batches with {len(all_run_results)} successful runs")
    
    return batch_results

def save_pro_anti_results(results, filename_prefix='pro_anti_classification'):
    """Save pro-anti results to files"""
    os.makedirs('paper4data', exist_ok=True)
    
    # Save detailed results
    with open(f'paper4data/{filename_prefix}_detailed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary DataFrame
    all_scores = []
    for topic, topic_results in results.items():
        for batch_result in topic_results:
            for comment, scores in batch_result['mean_scores'].items():
                row = {
                    'topic': topic,
                    'batch_id': batch_result['batch_id'],
                    'comment': comment[:200] + '...' if len(comment) > 200 else comment,
                    'comment_full': comment
                }
                # Add scores for each category (convert from 0-10 to 0-1 scale)
                for category in ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']:
                    score = scores.get(category, 0.0) / 10.0
                    row[category.lower().replace(' ', '_')] = score
                all_scores.append(row)
    
    if all_scores:
        # Write CSV with UTF-8 encoding
        fieldnames = ['topic', 'batch_id', 'comment', 'comment_full', 'pro', 'anti', 'pro_and_anti', 'neither']
        with open(f'paper4data/{filename_prefix}_scores.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)
        
        # Save summary statistics
        summary_stats = {}
        for topic in results.keys():
            topic_scores = []
            for batch_result in results[topic]:
                for comment, scores in batch_result['mean_scores'].items():
                    comment_scores = {}
                    for category in ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']:
                        score = scores.get(category, 0.0) / 10.0
                        comment_scores[category] = score
                    topic_scores.append(comment_scores)
            
            if topic_scores:
                # Calculate mean scores manually
                category_means = {}
                for category in ['PRO', 'ANTI', 'PRO AND ANTI', 'NEITHER']:
                    values = [score[category] for score in topic_scores if category in score]
                    if values:
                        category_means[category] = sum(values) / len(values)
                    else:
                        category_means[category] = 0.0
                
                summary_stats[topic] = {
                    'total_comments': len(topic_scores),
                    'mean_scores_by_category': category_means
                }
        
        # Write JSON with UTF-8 encoding
        with open(f'paper4data/{filename_prefix}_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to paper4data/{filename_prefix}_*")

def process_pro_anti_results_to_dataframe(results, sampled_comments=None):
    """Process pro-anti results into a DataFrame with proper formatting"""
    import pandas as pd
    
    rows = []
    for topic, topic_data in results.items():
        if topic_data and len(topic_data) > 0:
            for batch_data in topic_data:
                if 'mean_scores' in batch_data:
                    for comment_text, scores in batch_data['mean_scores'].items():
                        row = {
                            'comment_text': comment_text, 
                            'topic': topic,
                            'pro': scores.get('PRO', 0.0) / 10.0,
                            'anti': scores.get('ANTI', 0.0) / 10.0,
                            'pro_and_anti': scores.get('PRO AND ANTI', 0.0) / 10.0,
                            'neither': scores.get('NEITHER', 0.0) / 10.0
                        }
                        rows.append(row)
    
    # Create final DataFrame
    df = pd.DataFrame(rows)
    
    # Remove duplicates if any (keep the first occurrence which has the labels)
    df = df.drop_duplicates(subset=['comment_text', 'topic'], keep='first')
    
    print("DataFrame created with shape:", df.shape)
    print(f"Comments per topic:")
    for topic in df['topic'].unique():
        count = len(df[df['topic'] == topic])
        print(f"- {topic}: {count} comments")
    
    results_df = df.copy()
    for col in ['pro', 'anti', 'pro_and_anti', 'neither']:
        results_df[col] = results_df[col].astype(float)
    results_df = results_df.rename(columns={'comment_text': 'comment'})
    
    print("\nSample scores (should be between 0.0 and 1.0):")
    for topic in results_df['topic'].unique():
        topic_df = results_df[results_df['topic'] == topic]
        print(f"{topic}: {topic_df[['pro', 'anti', 'pro_and_anti', 'neither']].describe()}")
    
    return results_df

def classify_comments_pro_anti(comments, sector, num_runs=1, max_comments=None, num_agents=20, batch_size=10):
    """Classify comments into pro-anti categories for a specific sector/topic
    
    Args:
        comments: List of comments to classify
        sector: Sector name ('transport', 'housing', 'food')
        num_runs: Number of classification runs per batch (default: 1)
        max_comments: Maximum comments to process (default: None, process all)
        num_agents: Number of parallel agents to use (default: 20)
        batch_size: Number of comments per batch (default: 10)
    
    Returns:
        tuple: (topic_results, results_df)
    """
    # Check if sector is valid
    if sector not in SECTOR_TOPIC_MAPPING:
        print(f"Unknown sector: {sector}. Must be 'transport', 'housing', or 'food'.")
        return None, None
    
    # Get topic and hypotheses for the sector
    topic_info = SECTOR_TOPIC_MAPPING[sector]
    topic = topic_info['topic']
    hypotheses = topic_info['hypotheses']
    
    # Limit comments if specified
    if max_comments:
        comments = comments[:max_comments]
    
    print(f"Processing {sector} ({topic}) with {len(comments)} comments...")
    
    # Process the comments
    results = process_sector_comments_pro_anti(comments, hypotheses, topic, num_runs, num_agents, batch_size)
    
    # Save results for this topic
    topic_results = {topic: results}
    save_pro_anti_results(topic_results, filename_prefix=f'pro_anti_{sector}')
    
    # Process results to DataFrame
    results_df = process_pro_anti_results_to_dataframe(topic_results, {topic: comments})
    
    return topic_results, results_df

def main_pro_anti(comments_by_sector=None, num_runs=1, max_comments_per_sector=100, num_agents=20, batch_size=10):
    """Main function to process all sectors for pro-anti classification
    
    Args:
        comments_by_sector: Dict with sector names as keys and lists of comments as values
                           Format: {'transport': [comment1, comment2, ...], 
                                   'housing': [comment1, comment2, ...], 
                                   'food': [comment1, comment2, ...]}
        num_runs: Number of classification runs per batch (default: 1)
        max_comments_per_sector: Maximum comments to process per sector (default: 100)
        num_agents: Number of parallel agents to use (default: 20)
        batch_size: Number of comments per batch (default: 10)
    
    Returns:
        tuple: (results, results_df)
    """
    
    # Process each sector
    results = {}
    total_start_time = time.time()
    
    for sector in ['transport', 'housing', 'food']:
        comments = list(comments_by_sector.get(sector, []))[:max_comments_per_sector] if comments_by_sector else []
        if len(comments) > 0:
            print(f"\nProcessing {sector} sector with {len(comments)} comments...")
            sector_start_time = time.time()
            
            topic_results, _ = classify_comments_pro_anti(
                comments, 
                sector,
                num_runs,
                None,  # max_comments already handled above
                num_agents,
                batch_size
            )
            
            if topic_results:
                results.update(topic_results)
            
            sector_time = time.time() - sector_start_time
            print(f"{sector} completed in {sector_time:.2f} seconds")
        else:
            print(f"No comments found for {sector} sector")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Save combined results
    save_pro_anti_results(results, filename_prefix='pro_anti_all_sectors')
    
    # Process results to DataFrame
    results_df = process_pro_anti_results_to_dataframe(results, comments_by_sector)
    
    return results, results_df

def test_small_sample():
    """Test with a small sample of comments"""
    print("=== Testing Pro-Anti Classification with small sample ===")
    
    # Create a small test sample
    test_comments = [
        "I love my Tesla Model 3! The instant torque is amazing and it's so much fun to drive.",
        "Electric vehicles are terrible. They're too expensive and the range is awful.",
        "EVs are great for the environment but too expensive for most people.",
        "I went to the store and bought groceries today.",
        "Solar panels on my roof have really reduced my electricity bills.",
        "Solar farms are an eyesore and ruin the beautiful landscape.",
        "Solar power is good for the environment but takes up too much space.",
        "Going vegan has improved my health significantly.",
        "Vegan food tastes terrible and is too expensive.",
        "Veganism is ethical but the food is expensive and hard to find."
    ]
    
    print("\n--- Testing Transport (Electric Vehicles) ---")
    transport_results, transport_df = classify_comments_pro_anti(
        comments=test_comments,
        sector='transport',
        num_runs=3,
        num_agents=5,
        batch_size=5
    )
    
    print("\n--- Testing Housing (Solar Power) ---")
    housing_results, housing_df = classify_comments_pro_anti(
        comments=test_comments,
        sector='housing',
        num_runs=3,
        num_agents=5,
        batch_size=5
    )
    
    print("\n--- Testing Food (Vegetarianism/Veganism) ---")
    food_results, food_df = classify_comments_pro_anti(
        comments=test_comments,
        sector='food',
        num_runs=3,
        num_agents=5,
        batch_size=5
    )
    
    return transport_results, housing_results, food_results

if __name__ == "__main__":
    # Test with small sample
    test_small_sample() 