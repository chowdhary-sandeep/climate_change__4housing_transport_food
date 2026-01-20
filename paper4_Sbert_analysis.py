"""
This module provides functionality for semantic text analysis using SBERT (Sentence-BERT).
It includes functions for model setup, phrase similarity analysis, and visualization.
Modified for sector analysis (Paper 4).
"""

import os
import json
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import snapshot_download

def get_surrounding_words(text, pattern):
    """
    Extract surrounding words (3 words before and after) around a pattern in text.
    
    Args:
        text (str): The text to analyze
        pattern (str): The pattern to look for
        
    Returns:
        tuple: Lists of left and right context phrases
    """
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with','like','as'}
    
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    left_phrases, right_phrases = [], []
    pattern_words = pattern.split()
    pattern_len = len(pattern_words)
    
    for i in range(len(words) - pattern_len + 1):
        if words[i:i+pattern_len] == pattern_words:
            start_idx = max(0, i-3)
            left_context = words[start_idx:i]
            if left_context:
                left_phrases.append(' '.join(left_context))
            
            end_idx = min(len(words), i+pattern_len+3)
            right_context = words[i+pattern_len:end_idx]
            if right_context:
                right_phrases.append(' '.join(right_context))
            
    return left_phrases, right_phrases

def plot_curly_bracket_diagram(pattern, left_words, right_words, ax):
    """
    Plot a diagram showing word relationships with curly brackets.
    
    Args:
        pattern (str): The central keyword pattern
        left_words (dict): Words and percentages for left side
        right_words (dict): Words and percentages for right side
        ax (matplotlib.axes.Axes): The axis to plot on
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    words = pattern.split()
    text = '\n'.join(words)
    ax.text(5, 9, text, ha='center', va='center', fontsize=22, fontweight='bold',
            bbox=dict(facecolor='lightgray', edgecolor='black', pad=2))
    
    y_spacing = 8 / 4
    for i, (word, pct) in enumerate(list(left_words.items())[:3], 1):
        y = 9 - i * y_spacing
        color_intensity = 0.1 + 0.4 * (pct / max(left_words.values()))
        aged_yellow = (1, 0.95 - 0.2*color_intensity, 0.7 - 0.2*color_intensity)
        ax.text(1, y, f"{word}\n{pct:.1f}%", ha='left', va='center', fontsize=22,
                bbox=dict(facecolor=aged_yellow, edgecolor='none', alpha=0.4))
    
    for i, (word, pct) in enumerate(list(right_words.items())[:3], 1):
        y = 9 - i * y_spacing
        color_intensity = 0.1 + 0.4 * (pct / max(right_words.values()))
        professional_blue = (0.85 - 0.2*color_intensity, 0.9 - 0.2*color_intensity, 0.95 - 0.2*color_intensity) 
        ax.text(9, y, f"{word}\n{pct:.1f}%", ha='right', va='center', fontsize=22,
                bbox=dict(facecolor=professional_blue, edgecolor='none', alpha=0.4))
    
    ax.axis('off')

def setup_local_model(model_name='all-MiniLM-L6-v2'):
    """
    Download and setup the SBERT model locally.
    
    Args:
        model_name (str): Name of the model to download from sentence-transformers
        
    Returns:
        SentenceTransformer: Loaded model instance or None if setup fails
    """
    cache_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        local_model_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(local_model_path):
            print(f"Downloading model {model_name} to {local_model_path}...")
            snapshot_download(
                repo_id=f"sentence-transformers/{model_name}",
                local_dir=local_model_path,
                local_dir_use_symlinks=False
            )
        
        model = SentenceTransformer(local_model_path)
        return model
    
    except Exception as e:
        print(f"Error setting up local model: {e}")
        return None

def merge_similar_phrases_sbert(word_counts, pattern, is_left=True, similarity_threshold=0.7):
    """
    Merge similar phrases using SBERT embeddings and cosine similarity.
    
    Args:
        word_counts (Counter): Dictionary of word counts
        pattern (str): The keyword pattern being analyzed
        is_left (bool): Whether these are left context words
        similarity_threshold (float): Threshold for considering phrases similar
        
    Returns:
        dict: Merged word counts with similar phrases combined
    """
    model = setup_local_model()
    if model is None:
        print("Failed to load model, returning original word counts")
        return word_counts

    merged = {}
    phrases = list(word_counts.keys())
    used = set()

    valid_phrases = []
    valid_counts = {}
    for p in phrases:
        if len(p) > 2 and p.replace('.','').replace(',','').replace("'", "").replace(" ", "").isalnum():
            valid_phrases.append(p)
            valid_counts[p] = word_counts[p]
    
    if not valid_phrases:
        return word_counts

    batch_size = 64
    full_phrases = [p + ' ' + pattern if is_left else pattern + ' ' + p for p in valid_phrases]
    embeddings = []

    for i in range(0, len(full_phrases), batch_size):
        batch = full_phrases[i:i+batch_size]
        try:
            batch_embeddings = model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error encoding batch {i//batch_size}: {e}")
            continue
            
    if not embeddings:
        return word_counts
        
    embeddings = np.vstack(embeddings)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    similarities = cosine_similarity(embeddings)

    for i, phrase1 in enumerate(valid_phrases):
        if phrase1 in used:
            continue
            
        total = valid_counts[phrase1]
        similar_phrases = [phrase1]
        used.add(phrase1)
        
        for j, phrase2 in enumerate(valid_phrases[i+1:], i+1):
            if phrase2 not in used and similarities[i,j] >= similarity_threshold:
                total += valid_counts[phrase2]
                similar_phrases.append(phrase2)
                used.add(phrase2)
                
        rep_phrase = max(similar_phrases, key=lambda x: valid_counts[x])
        merged[rep_phrase] = total

    for p in word_counts:
        if p not in used:
            merged[p] = word_counts[p]

    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True)[:3])

def analyze_keyword_contexts(keyword_comment_map, sector_keyword_strength):
    """
    Analyze keyword contexts and generate visualizations.
    
    Args:
        keyword_comment_map (dict): Mapping of keywords to their comment contexts
        sector_keyword_strength (dict): Classification of keywords by sector and strength
        
    Returns:
        tuple: (keyword_context dict, strong_keywords_by_volume list)
    """
    keyword_context = {}
    left_data = []
    right_data = []
    strong_keywords_by_volume = []

    # Create a flat list of all strong keywords from all sectors
    all_strong_keywords = []
    for category in sector_keyword_strength.keys():
        if category.endswith('_strong'):
            all_strong_keywords.extend(sector_keyword_strength[category])
    
    # Create a mapping from keyword to sector
    keyword_to_sector = {}
    for category, keywords in sector_keyword_strength.items():
        if category.endswith('_strong') or category.endswith('_weak'):
            sector = category.replace('_strong', '').replace('_weak', '')
            for keyword in keywords:
                keyword_to_sector[keyword] = sector

    # Filter keywords that meet the threshold
    keywords_to_process = [k for k in keyword_comment_map.keys() 
                         if len(keyword_comment_map[k]['matches']) >= 200]
    
    total_keywords = len(keywords_to_process)
    num_figures = math.ceil(total_keywords / 9)
    figures = []
    axes = []

    # Create figures with 30% wider width
    for i in range(num_figures):
        fig, ax = plt.subplots(3, 3, figsize=(26, 18))  # Increased from 20 to 26
        figures.append(fig)
        axes.append(ax)

    start_time = time.time()
    
    for idx, pattern in enumerate(tqdm(keywords_to_process, desc="Processing keywords", 
                                     unit="keyword", total=total_keywords)):
        left_counter = Counter()
        right_counter = Counter()
        
        for comment in keyword_comment_map[pattern]['matches']:
            left, right = get_surrounding_words(comment.replace('\n', ' '), pattern.lower())
            left_counter.update(left)
            right_counter.update(right)
        
        # Filter out invalid entries (increased from 100 to 200)
        left_counter = Counter({k: v for k, v in left_counter.most_common(500) 
                              if k.strip() and 'http' not in k})
        right_counter = Counter({k: v for k, v in right_counter.most_common(500) 
                               if k.strip() and 'http' not in k})
        
        if not left_counter or not right_counter:
            continue
        
        left_total = sum(left_counter.values())
        right_total = sum(right_counter.values())
        
        left_merged = merge_similar_phrases_sbert(left_counter, pattern, True)
        right_merged = merge_similar_phrases_sbert(right_counter, pattern, False)
        
        left_words = {w: c/left_total*100 for w,c in left_merged.items()}
        right_words = {w: c/right_total*100 for w,c in right_merged.items()}
        
        # Get sector information for this keyword
        sector = keyword_to_sector.get(pattern, 'unknown')
        
        left_data.extend([{'keyword': pattern, 'sector': sector, 'word': w, 'percentage': p} 
                         for w,p in left_words.items()])
        right_data.extend([{'keyword': pattern, 'sector': sector, 'word': w, 'percentage': p} 
                          for w,p in right_words.items()])
        
        if pattern in all_strong_keywords:
            strong_keywords_by_volume.append((pattern, len(keyword_comment_map[pattern]['matches']), sector))
        
        fig_num = idx // 9
        subplot_idx = idx % 9
        ax = axes[fig_num][subplot_idx//3, subplot_idx%3]
        
        plot_curly_bracket_diagram(pattern, left_words, right_words, ax)
        keyword_context[pattern] = {'left': left_words, 'right': right_words, 'sector': sector}

    # Save all data
    os.makedirs('paper4data', exist_ok=True)
    os.makedirs('paper4figs', exist_ok=True)
    
    pd.DataFrame(left_data).to_csv('paper4data/keyword_context_left_local.csv', index=False)
    pd.DataFrame(right_data).to_csv('paper4data/keyword_context_right_local.csv', index=False)
    with open('paper4data/keyword_context_local.json', 'w') as f:
        json.dump(keyword_context, f)

    # Save all figures
    for i, fig in enumerate(figures):
        plt.figure(fig.number)
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=1.0)
        fig.savefig(f'paper4figs/figSI_{i+1}_local.pdf', dpi=300, bbox_inches='tight')
        fig.savefig(f'paper4figs/figSI_{i+1}_local.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
    return keyword_context, strong_keywords_by_volume

def create_sector_keyword_strength():
    """
    Create the sector keyword strength structure for Paper 4 analysis.
    
    Returns:
        dict: Sector keyword strength classification
    """
    sector_keyword_strength = {

        # --------------------------- 1. TRANSPORT  ---------------------------
        # focus: electric vehicles (EVs) and low-carbon mobility
        'transport_strong': [
            # electric-vehicle needle-in-a-haystack words
            "ev", "electric vehicle", "evs", "bev", "battery electric",
            "tesla model", "model 3", "model y", "chevy bolt", "nissan leaf",
            "ioniq 5", "mustang mach-e", "id.4", "rivian", "lucid air",
            "supercharger", "gigafactory",
            # charging infrastructure & policy triggers
            "level 2 charger", "dc fast charger", "public charger", "home charger",
            "charging network", "range anxiety", "mpge",
            # mode-shift / car-free framing
            "bike lane", "protected cycleway", "car-free", "low emission zone"
        ],

        'transport_weak': [
            # generic EV talk
            "electric car", "electric truck", "electric suv", "plug-in hybrid",
            "phev", "charging station", "charge point", "kw charger",
            "battery swap", "solid-state battery", "gigacast",
            # policy, subsidy, mandates
            "tax credit", "zev mandate", "ev rebate", "incentive", "phase-out ice",
            # complementary green mobility
            "e-bike", "micro-mobility", "last-mile delivery", "transit electrification",
            # brand spill-over
            "tesla", "spacex launch price?", "elon says",   # keeps tesla chatter
            # broader transport decarb
            "rail electrification", "hydrogen truck", "low carbon transport"
        ],

        # --------------------------- 2. HOUSING / RESIDENTIAL ENERGY  ---------------------------
        # focus: rooftop solar, PV, home efficiency, heat pumps
        'housing_strong': [
            "rooftop solar", "solar pv", "pv panel", "photovoltaics",
            "solar array", "net metering", "feed-in tariff", "solar inverter",
            "kwh generated", "solar roof", "sunrun", "sunpower",
            # electrification add-ons
            "heat pump", "air-source heat pump", "ground-source heat pump",
            "mini-split", "building retrofit", "home insulation", "passivhaus"
        ],

        'housing_weak': [
            "solar panel", "solar panels", "solar power", "solar installer",
            "battery storage", "powerwall", "home battery", "smart thermostat",
            "energy audit", "energy efficiency upgrade", "led retrofit",
            "green home", "net-zero house", "zero-energy building",
            # policy and finance
            "solar tax credit", "pvgis", "renewable portfolio standard",
            "community solar", "virtual power plant", "rooftop rebate"
        ],

        # --------------------------- 3. FOOD  ---------------------------
        # focus: veganism vs. meat eating / reduction
        'food_strong': [
            # explicit vegan / vegetarian identifiers
            "vegan", "plant-based diet", "plant based", "veganism",
            "veganuary", "vegetarian", "veg lifestyle",
            # meat-centric phrases
            "carnivore diet", "meat lover", "steakhouse", "barbecue festival",
            "bacon double", "grass-fed beef", "factory farming",
            # policy / campaign cues
            "meatless monday", "beyond meat", "impossible burger",
            "plant-based burger", "animal cruelty free"
        ],

        'food_weak': [
            # generic protein or flex-terms
            "red meat", "beef consumption", "dairy free", "plant protein",
            "soy burger", "nutritional yeast", "seitan", "tofurky",
            # emissions framing
            "agricultural emissions", "methane footprint", "carbon hoofprint",
            "cow burps", "livestock emissions", "feedlot",
            # lifestyle / cooking talk
            "recipe vegan", "tofu scramble", "almond milk", "oat milk",
            "flexitarian", "climatetarian",
            # alternative proteins
            "cultivated meat", "lab-grown meat", "precision fermentation"
        ]
    }
    
    return sector_keyword_strength 