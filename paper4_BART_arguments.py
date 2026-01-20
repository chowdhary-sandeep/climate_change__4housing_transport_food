"""
Sentence-BERT (SBERT) Classification for Reddit Comments
Based on paper4_GPT_arguments.py but using local SBERT models with BART as optional alternative
"""

import os
import time
import json
import pickle
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
import threading
import numpy as np
import logging
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re # Added for hypothesis example extraction

# Import sentence transformers for SBERT
try:
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers library not found. Install with: pip install sentence-transformers")

# Import transformers for BART zero-shot classification (optional)
try:
    from transformers import pipeline
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers torch")

def setup_local_model(model_name='all-MiniLM-L6-v2'):
    """
    Download and setup the SBERT model locally.
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device set to use {device}")
        
        model = SentenceTransformer(local_model_path, device=device)
        return model
    
    except Exception as e:
        logging.error(f"Error setting up local model: {e}")
        return None

# ----------------------------  TRANSPORT  ---------------------------- #
transport_hypotheses = {
    # --- existing tags ---                                                      #
    "Battery Cost": (
        "Mentions replacement or recycling cost of EV batteries, battery warranties, "
        "or battery prices per kWh.  Cues: 'battery pack', 'replacement cost', "
        "'$ / kWh', 'warranty', 'recycle'."
    ),
    "Purchase Price": (
        "Discusses up-front sticker price, MSRP, subsidies, or tax credits for buying an EV. "
        "Cues: 'purchase price', 'MSRP', 'tax rebate', 'federal credit', 'too expensive to buy'."
    ),
    "Range Anxiety": (
        "Expresses fear or inconvenience over limited driving range or long-distance travel. "
        "Cues: 'range anxiety', 'miles per charge', 'can't make the trip', 'stops to charge'."
    ),
    "Charging Infrastructure": (
        "Talks about availability, speed, or reliability of public or home chargers. "
        "Cues: 'chargers', 'supercharger', 'fast charge', 'Level 2', 'installation'."
    ),
    "Environmental Benefit": (
        "Claims EVs reduce emissions or pollution vs. gasoline cars. "
        "Cues: 'carbon footprint', 'emissions', 'cleaner', 'zero tailpipe'."
    ),
    "Running Costs": (
        "Mentions electricity cost, fuel savings, maintenance savings or total cost of ownership. "
        "Cues: '$ per mile', 'electricity bill', 'maintenance cost', 'cheaper to run'."
    ),
    "Reliability": (
        "Questions or praises mechanical reliability, battery life, or durability of EVs. "
        "Cues: 'reliability', 'breakdown', 'battery degradation', 'repair rates'."
    ),
    "Fun to Drive": (
        "Highlights acceleration, torque, quietness, or general driving pleasure of EVs. "
        "Cues: 'instant torque', '0-60', 'fun to drive', 'performance'."
    ),
    # --- emergent extras you already added ---                                   #
    "Safety & Fire Risk": (
        "Raises concerns about EV-related fires, thermal runaway, unintended braking or autopilot crashes. "
        "Cues: 'battery fire', 'thermal runaway', 'phantom braking', 'autopilot crash'."
    ),
    "Battery Recycling & End-of-Life": (
        "Discusses second-life battery uses, recycling logistics, 'bricking' or disposal challenges. "
        "Cues: 'end-of-life', 'reuse powerwall', 'battery bricks', 'recycle program'."
    ),
    "Equity & Accessibility": (
        "Points out that EV adoption favors wealthier buyers, debates subsidy fairness for lower-income users. "
        "Cues: 'low-income', 'wealth gap', 'grant program', 'accessibility', 'subsidy'."
    ),
    # --- NEW coverage tags ---                                                   #
    "Mineral Supply Chain": (
        "Concerns over lithium, cobalt, nickel, rare-earth mining or shortages for batteries. "
        "Cues: 'lithium shortage', 'cobalt supply', 'critical minerals', 'mine', 'geopolitics'."
    ),
    "Policy & Mandates": (
        "References government regulations, bans on ICE sales, fleet targets or central-planning critiques. "
        "Cues: '2035 ban', 'zero-emission mandate', 'EPA rule', 'central planning', 'compliance'."
    ),
    "Lifecycle Emissions": (
        "Talks cradle-to-grave or well-to-wheel carbon footprint of EVs vs. gas vehicles. "
        "Cues: 'lifecycle CO2', 'embedded emissions', 'production footprint', 'total carbon budget'."
    ),
    "Depreciation & Resale": (
        "Mentions resale value, depreciation risk, leasing to hedge EV uncertainty. "
        "Cues: 'depreciation', 'resale', 'lease deal', 'trade-in value'."
    ),
    "Insurance & Ownership Risk": (
        "Insurance premiums, accident repair cost, battery replacement after crash. "
        "Cues: 'insurance cost', 'write-off', 'totaled', 'premium hike'."
    ),
    "Alternative Modes": (
        "Advocates bikes, transit, e-scooters, trains instead of private EVs. "
        "Cues: 'public transit', 'bike lane', 'micro-mobility', 'train over car'."
    ),
    "Second-Hand Market": (
        "Talks about used EV prices, battery warranty transfer, or buying older models. "
        "Cues: 'used Leaf', 'second-hand Model 3', 'certified pre-owned', 'battery health report'."
    ),
    "Grid Impact & Energy Mix": (
        "Links EV charging to grid capacity, blackout fears, renewable share of electricity. "
        "Cues: 'grid strain', 'brownout', 'coal-powered EV', 'smart charging', 'vehicle-to-grid'."
    ),
    "Technology Roadmap": (
        "Mentions solid-state batteries, super-caps, hydrogen fuel-cell cars, or other next-gen tech. "
        "Cues: 'solid-state', 'Li-S', 'fuel-cell', 'graphene', 'fast-charging breakthrough'."
    ),
}

# ----------------------------  HOUSING / RENEWABLES  ---------------------------- #
housing_hypotheses = {
    # --- existing tags ---                                                      #
    "Local Economy": (
        "Claims solar/wind projects create or harm local jobs, investment, or economic growth. "
        "Cues: 'local jobs', 'investment', 'economic boost', 'tourism', 'industry'."
    ),
    "Utility Bills": (
        "Mentions household or community electricity bills going up or down due to solar/wind. "
        "Cues: 'lower bills', 'savings', '$/kWh', 'feed-in tariff', 'payback'."
    ),
    "Visual Impact": (
        "Talks about aesthetics, view-scape, or ugliness/beauty of solar panels or farms. "
        "Cues: 'eyesore', 'ruin the view', 'aesthetic', 'visual impact'."
    ),
    "Land Use": (
        "Raises land-area or space requirements, farmland loss, or siting footprint of solar/wind. "
        "Cues: 'acres', 'square miles', 'farmland', 'competes with agriculture'."
    ),
    "Tax Revenue": (
        "Mentions local taxes or revenue streams from solar/wind projects. "
        "Cues: 'tax revenue', 'levy', 'rates base', 'municipal income'."
    ),
    "Wildlife": (
        "Discusses harm or benefit to wildlife, habitat, or biodiversity from solar/wind. "
        "Cues: 'habitat', 'birds', 'biodiversity', 'ecosystem', 'wildlife corridor'."
    ),
    # --- your extras ---                                                        #
    "Cost Volatility & Economic Risk": (
        "Highlights sudden spikes in panel/turbine prices, CAPEX uncertainty, ROI fears. "
        "Cues: 'price surge', 'cost spike', 'expensive panels', 'ROI not pencil'."
    ),
    "Green Jobs & Workforce": (
        "Notes creation, training or loss of jobs in renewable installation and related industries. "
        "Cues: 'green jobs', 'installer', 'training', 'workforce gap'."
    ),
    "Subsidy & Tariff Debate": (
        "Argues over feed-in-tariffs, net-metering rules or subsidy fairness. "
        "Cues: 'net metering', 'FIT', 'subsidy', 'tax credit', 'ratepayer'."
    ),
    # --- NEW coverage tags ---                                                  #
    "Grid Stability & Storage": (
        "Discussions of intermittency, batteries, pumped hydro, or grid reliability with high renewables. "
        "Cues: 'intermittent', 'storage', 'backup', 'blackout', 'grid stability'."
    ),
    "Permitting & Bureaucracy": (
        "Complaints about slow permits, local opposition meetings, environmental impact statements. "
        "Cues: 'red tape', 'zoning', 'EIS', 'planning board', 'permit delay'."
    ),
    "Property Value Impact": (
        "Claims renewable projects raise or lower nearby property values. "
        "Cues: 'property values', 'home price', 'real-estate depreciation'."
    ),
    "Noise (Wind)": (
        "Mentions turbine noise, infrasonic health worries, or sleep disturbance. "
        "Cues: 'whooshing', 'infrasound', 'noise pollution', 'shadow flicker'."
    ),
    "Decommissioning & Waste": (
        "Talks about end-of-life panel/turbine disposal, recycling, landfill issues. "
        "Cues: 'blade landfill', 'panel recycling', 'decommission', 'waste stream'."
    ),
    "Foreign Dependence & Trade": (
        "References Chinese panel dominance, tariffs, trade wars, or reshoring supply chains. "
        "Cues: 'Chinese solar', 'import duty', 'trade dispute', 'domestic manufacturing'."
    ),
    "Community Engagement & NIMBY": (
        "Local meetings, petitions, or protests opposing nearby renewable builds. "
        "Cues: 'NIMBY', 'community meeting', 'public hearing', 'lawsuit', 'moratorium'."
    ),
    "Resilience & Blackout Protection": (
        "Frames solar+storage as protection against outages or storms; or critiques reliability. "
        "Cues: 'backup power', 'resilience hub', 'blackout', 'micro-grid'."
    ),
}

# ----------------------------  FOOD / DIET  ---------------------------- #
food_hypotheses = {
    # --- existing tags ---                                                      #
    "Health": (
        "Claims physical health benefits or risks of eating less meat / going vegan. "
        "Cues: 'blood pressure', 'cholesterol', 'healthy', 'nutrients'."
    ),
    "Environmental Impact": (
        "Links diet choice to climate change, land, water, or emissions. "
        "Cues: 'carbon', 'methane', 'land footprint', 'water use'."
    ),
    "Animal Welfare": (
        "Cites animal suffering, cruelty, or ethics as motivation. "
        "Cues: 'cruelty', 'slaughter', 'factory farming', 'sentient'."
    ),
    "Cost": (
        "Mentions food affordability, grocery bills, or relative prices of meat vs. plant foods. "
        "Cues: 'cheaper', 'expensive', '$ per pound', 'budget'."
    ),
    "Taste & Convenience": (
        "Talks about flavour, texture, cooking ease, availability of vegan options, or social convenience. "
        "Cues: 'tastes good', 'bland', 'easy recipe', 'prep time'."
    ),
    # --- your extras ---                                                        #
    "Systemic vs Individual Action": (
        "Calls for policy, corporate reform or large-scale funding instead of just personal diet shifts. "
        "Cues: 'policy change', 'meat tax', 'corporate responsibility'."
    ),
    "Cultural Identity & Tradition": (
        "Invokes heritage foods, family customs or regional cuisine as barrier/motivation. "
        "Cues: 'traditional dish', 'family recipe', 'cultural staple'."
    ),
    "Health Deficiency Anxiety": (
        "Concerns over nutrient shortfalls, supplements or medical monitoring on vegan diets. "
        "Cues: 'deficiency', 'B12', 'protein gap', 'doctor said'."
    ),
    # --- NEW coverage tags ---                                                  #
    "Lab-Grown & Alt Proteins": (
        "References cultivated meat, precision fermentation, insect protein or plant-based substitutes. "
        "Cues: 'lab-grown', 'cultivated chicken', 'Beyond Burger', 'mycoprotein'."
    ),
    # "Food Security & Access": (
    #     "Discusses global hunger, supply shocks, price volatility, or equitable access to protein. "
    #     "Cues: 'food insecurity', 'ration', 'supply chain', 'hunger prices'."
    # ),
    "Farmer Livelihoods": (
        "Impacts on ranchers, smallholders, or crop growers from shifting diets. "
        "Cues: 'farm income', 'livestock jobs', 'dairy bailout', 'subsidy shift'."
    ),
    "Social Media Influence": (
        "Role of influencers, documentaries, viral challenges shaping diet choices. "
        "Cues: 'What the Health', 'Netflix doc', 'TikTok vegan', 'influencer recipe'."
    ),
    "Psychology & Identity": (
        "Diet as part of personal identity, moral virtue signalling or tribal politics. "
        "Cues: 'vegan pride', 'carnivore tribe', 'identity', 'virtue signaling'."
    ),
}

def get_hypothesis_examples(hypothesis_name, hypothesis_desc):
    """Extract key phrases and examples from hypothesis description"""
    # Extract phrases in quotes as examples
    examples = re.findall(r"'([^']*)'", hypothesis_desc)
    # Add the hypothesis name itself as a phrase
    key_phrases = [hypothesis_name.lower()] + [ex.lower() for ex in examples]
    # Remove duplicates while preserving order
    return list(dict.fromkeys(key_phrases))

class SBERTClassifier:
    """SBERT Classifier using semantic similarity with key phrases"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None):
        if not SBERT_AVAILABLE:
            raise ImportError("sentence-transformers library required")
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self._lock = threading.Lock()
        
    def _initialize_model(self):
        """Initialize the model (lazy loading)"""
        if self.model is None:
            with self._lock:
                if self.model is None:
                    print(f"Loading SBERT model: {self.model_name}")
                    self.model = setup_local_model(self.model_name)
                    if self.model is None:
                        raise RuntimeError(f"Failed to load SBERT model {self.model_name}")
                    print("✓ SBERT model loaded successfully")
    
    def classify_batch_hypotheses(self, comments_batch, hypotheses, batch_size=50):
        """
        Classify comments using semantic similarity with key phrases from hypotheses
        """
        self._initialize_model()
        
        # Extract key phrases for each hypothesis
        hypothesis_phrases = {
            name: get_hypothesis_examples(name, desc)
            for name, desc in hypotheses.items()
        }
        
        # Prepare all unique phrases for embedding
        all_phrases = []
        phrase_to_hypothesis = {}
        for hyp_name, phrases in hypothesis_phrases.items():
            for phrase in phrases:
                all_phrases.append(phrase)
                phrase_to_hypothesis[phrase] = hyp_name
        
        # Get phrase embeddings (once for all comments)
        phrase_embeddings = self.model.encode(all_phrases, show_progress_bar=False)
        
        all_results = {}
        
        # Process comments in batches
        for i in range(0, len(comments_batch), batch_size):
            batch_comments = comments_batch[i:i + batch_size]
            
            # Get comment embeddings
            comment_embeddings = self.model.encode(batch_comments, show_progress_bar=False)
            
            # Calculate similarities
            similarities = cosine_similarity(comment_embeddings, phrase_embeddings)
            
            # Process results for each comment
            for j, comment in enumerate(batch_comments):
                # Group similarities by hypothesis
                hypothesis_scores = defaultdict(list)
                for k, phrase in enumerate(all_phrases):
                    hyp_name = phrase_to_hypothesis[phrase]
                    similarity = similarities[j][k]
                    hypothesis_scores[hyp_name].append(similarity)
                
                # Take max similarity for each hypothesis
                comment_results = {}
                for hyp_name, scores in hypothesis_scores.items():
                    max_score = max(scores)  # Best match among phrases
                    confidence = (max_score + 1) / 2.0  # Convert to 0-1 scale
                    
                    comment_results[hyp_name] = {
                        'score': confidence,
                        'relevant': confidence >= 0.5,
                        'confidence': confidence,
                        'raw_similarity': max_score
                    }
                
                all_results[comment] = comment_results
        
        return all_results

class BARTClassifier:
    """BART Zero-Shot Classifier for hypothesis classification"""
    
    def __init__(self, model_name="facebook/bart-large-mnli", device=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device
        self.classifier = None
        self._lock = threading.Lock()
        
    def _initialize_model(self):
        """Initialize the model (lazy loading)"""
        if self.classifier is None:
            with self._lock:
                if self.classifier is None:
                    print(f"Loading BART model: {self.model_name}")
                    self.classifier = pipeline(
                        "zero-shot-classification",
                        model=self.model_name,
                        device=self.device
                    )
                    print("✓ BART model loaded successfully")
    
    def classify_single_hypothesis(self, comment, hypothesis_name, hypothesis_description):
        """
        Classify a single comment against a single hypothesis using BART zero-shot
        
        Args:
            comment: Text to classify
            hypothesis_name: Name of the hypothesis 
            hypothesis_description: Description of what the hypothesis looks for
            
        Returns:
            dict: {'hypothesis': name, 'score': confidence_score, 'label': 'relevant'/'not_relevant'}
        """
        self._initialize_model()
        
        # Create labels for zero-shot classification
        labels = [f"relevant to {hypothesis_name}", "not relevant"]
        
        # Create a more detailed prompt using the hypothesis description
        prompt = f"Comment: {comment}\n\nContext: {hypothesis_description}"
        
        try:
            # Classify using BART
            result = self.classifier(prompt, labels)
            
            # Extract relevant score
            relevant_label = f"relevant to {hypothesis_name}"
            if result['labels'][0] == relevant_label:
                confidence = result['scores'][0]
                label = 'relevant'
            else:
                confidence = result['scores'][1] if len(result['scores']) > 1 else 1 - result['scores'][0]
                label = 'not_relevant'
            
            return {
                'hypothesis': hypothesis_name,
                'score': confidence,
                'label': label,
                'raw_result': result
            }
            
        except Exception as e:
            print(f"Error classifying comment for {hypothesis_name}: {e}")
            return {
                'hypothesis': hypothesis_name,
                'score': 0.0,
                'label': 'not_relevant',
                'error': str(e)
            }

def classify_comment_all_hypotheses(comment, hypotheses, sector, classifier, threshold=0.5):
    """
    Classify a single comment against all hypotheses in a sector
    
    Args:
        comment: Text to classify
        hypotheses: Dictionary of hypothesis_name -> description
        sector: Sector name
        classifier: SBERTClassifier or BARTClassifier instance
        threshold: Minimum confidence threshold for positive classification
        
    Returns:
        dict: Results for all hypotheses
    """
    results = {}
    
    for hypothesis_name, hypothesis_desc in hypotheses.items():
        result = classifier.classify_single_hypothesis(comment, hypothesis_name, hypothesis_desc)
        
        # Convert to binary classification based on threshold
        is_relevant = result['score'] >= threshold and result['label'] == 'relevant'
        
        results[hypothesis_name] = {
            'score': result['score'],
            'relevant': is_relevant,
            'confidence': result['score']
        }
    
    return results

def process_comments_batch_sbert_efficient(comments_batch, hypotheses, sector, batch_id, classifier, batch_size=50):
    """Process a batch of comments using efficient SBERT batch processing"""
    print(f"Processing {sector} batch {batch_id} with {len(comments_batch)} comments using SBERT")
    
    # Use the efficient batch processing method
    batch_results = classifier.classify_batch_hypotheses(comments_batch, hypotheses, batch_size)
    
    # Convert to the expected format
    formatted_results = []
    for i, comment in enumerate(comments_batch):
        comment_id = f"comment_{batch_id}_{i+1}"
        
        formatted_results.append({
            'comment_id': comment_id,
            'comment': comment,
            'hypothesis_results': batch_results[comment]
        })
    
    return {
        'sector': sector,
        'batch_id': batch_id,
        'comments': comments_batch,
        'results': formatted_results,
        'hypotheses': list(hypotheses.keys())
    }

def process_sector_comments_sbert(comments, hypotheses, sector, num_runs=10, num_workers=4, batch_size=10, sbert_batch_size=50):
    """
    Process all comments for a sector using efficient SBERT classification with multiple runs
    
    Args:
        comments: List of comments to classify
        hypotheses: Dictionary of hypothesis_name -> description
        sector: Sector name
        num_runs: Number of classification runs per batch
        num_workers: Number of parallel workers
        batch_size: Number of comments per processing batch
        sbert_batch_size: Internal batch size for SBERT encoding (default: 50)
    """
    print(f"Processing {len(comments)} comments for {sector} with SBERT")
    print(f"Configuration: {num_runs} runs per batch, {num_workers} workers, batch size {batch_size}, SBERT batch size {sbert_batch_size}")
    
    # Initialize classifier
    classifier = SBERTClassifier()
    
    # Split comments into batches
    comment_batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]
    
    # Create tasks for multiple runs
    all_results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Process batches in parallel for this run
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for batch_id, batch_comments in enumerate(comment_batches):
                future = executor.submit(
                    process_comments_batch_sbert_efficient,
                    batch_comments,
                    hypotheses,
                    sector,
                    f"{batch_id+1}_run{run+1}",
                    classifier,
                    sbert_batch_size
                )
                futures.append(future)
            
            # Collect results with progress bar
            batch_results = []
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"SBERT {sector} run {run+1}"):
                result = future.result()
                batch_results.append(result)
        
        all_results.extend(batch_results)
    
    # Calculate mean scores across runs
    print("Calculating mean scores across runs...")
    comment_scores = defaultdict(lambda: defaultdict(list))
    
    # Collect all scores for each comment-hypothesis pair
    for batch_result in all_results:
        for comment_result in batch_result['results']:
            comment = comment_result['comment']
            for hypothesis, result in comment_result['hypothesis_results'].items():
                comment_scores[comment][hypothesis].append(result['score'])
    
    # Calculate means
    mean_scores = {}
    for comment, hypothesis_scores in comment_scores.items():
        mean_scores[comment] = {}
        for hypothesis, scores in hypothesis_scores.items():
            mean_scores[comment][hypothesis] = np.mean(scores) * 10.0  # Scale to 0-10 for compatibility
    
    # Package final results
    final_results = [{
        'sector': sector,
        'batch_id': 'combined',
        'comments': comments,
        'mean_scores': mean_scores,
        'hypotheses': list(hypotheses.keys()),
        'raw_results': all_results
    }]
    
    return final_results

def process_comments_batch_bart(comments_batch, hypotheses, sector, batch_id, classifier, threshold=0.5):
    """Process a batch of comments using BART classification"""
    print(f"Processing {sector} batch {batch_id} with {len(comments_batch)} comments using BART")
    
    batch_results = []
    
    for i, comment in enumerate(comments_batch):
        comment_id = f"comment_{batch_id}_{i+1}"
        
        # Classify against all hypotheses
        hypothesis_results = classify_comment_all_hypotheses(
            comment, hypotheses, sector, classifier, threshold
        )
        
        batch_results.append({
            'comment_id': comment_id,
            'comment': comment,
            'hypothesis_results': hypothesis_results
        })
    
    return {
        'sector': sector,
        'batch_id': batch_id,
        'comments': comments_batch,
        'results': batch_results,
        'hypotheses': list(hypotheses.keys())
    }

def process_sector_comments_bart(comments, hypotheses, sector, num_runs=10, num_workers=4, batch_size=10, threshold=0.5):
    """
    Process all comments for a sector using BART classification with multiple runs
    
    Args:
        comments: List of comments to classify
        hypotheses: Dictionary of hypothesis_name -> description
        sector: Sector name
        num_runs: Number of classification runs per batch
        num_workers: Number of parallel workers
        batch_size: Number of comments per batch
        threshold: Classification threshold
    """
    print(f"Processing {len(comments)} comments for {sector} with BART")
    print(f"Configuration: {num_runs} runs per batch, {num_workers} workers, batch size {batch_size}")
    
    # Initialize classifier
    classifier = BARTClassifier()
    
    # Split comments into batches
    comment_batches = [comments[i:i + batch_size] for i in range(0, len(comments), batch_size)]
    
    # Create tasks for multiple runs
    all_results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run + 1}/{num_runs} ---")
        
        # Process batches in parallel for this run
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for batch_id, batch_comments in enumerate(comment_batches):
                future = executor.submit(
                    process_comments_batch_bart,
                    batch_comments,
                    hypotheses,
                    sector,
                    f"{batch_id+1}_run{run+1}",
                    classifier,
                    threshold
                )
                futures.append(future)
            
            # Collect results with progress bar
            batch_results = []
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc=f"BART {sector} run {run+1}"):
                result = future.result()
                batch_results.append(result)
        
        all_results.extend(batch_results)
    
    # Calculate mean scores across runs
    print("Calculating mean scores across runs...")
    comment_scores = defaultdict(lambda: defaultdict(list))
    
    # Collect all scores for each comment-hypothesis pair
    for batch_result in all_results:
        for comment_result in batch_result['results']:
            comment = comment_result['comment']
            for hypothesis, result in comment_result['hypothesis_results'].items():
                comment_scores[comment][hypothesis].append(result['score'])
    
    # Calculate means
    mean_scores = {}
    for comment, hypothesis_scores in comment_scores.items():
        mean_scores[comment] = {}
        for hypothesis, scores in hypothesis_scores.items():
            mean_scores[comment][hypothesis] = np.mean(scores) * 10.0  # Scale to 0-10 for compatibility
    
    # Package final results
    final_results = [{
        'sector': sector,
        'batch_id': 'combined',
        'comments': comments,
        'mean_scores': mean_scores,
        'hypotheses': list(hypotheses.keys()),
        'raw_results': all_results
    }]
    
    return final_results

def save_results_sbert(results, filename_prefix='SBERT_arguments'):
    """Save SBERT classification results"""
    os.makedirs('paper4data', exist_ok=True)
    
    # Save detailed results
    with open(f'paper4data/{filename_prefix}_detailed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary CSV
    all_scores = []
    all_hypotheses = set()
    
    for sector, sector_results in results.items():
        for batch_result in sector_results:
            if 'mean_scores' in batch_result:
                all_hypotheses.update(batch_result['hypotheses'])
                
                for comment, scores in batch_result['mean_scores'].items():
                    row = {
                        'sector': sector,
                        'comment': comment[:200] + '...' if len(comment) > 200 else comment,
                        'comment_full': comment
                    }
                    
                    for hypothesis in batch_result['hypotheses']:
                        clean_hypothesis = hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
                        score = scores.get(hypothesis, 0.0) / 10.0  # Convert to 0-1 scale
                        row[clean_hypothesis] = score
                    
                    all_scores.append(row)
    
    # Save CSV
    if all_scores:
        import csv
        clean_hypotheses = [h.lower().replace(' ', '_').replace('&', 'and').replace('-', '_') for h in sorted(all_hypotheses)]
        fieldnames = ['sector', 'comment', 'comment_full'] + clean_hypotheses
        
        with open(f'paper4data/{filename_prefix}_scores.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)
    
    # Save JSON summary
    with open(f'paper4data/{filename_prefix}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ SBERT results saved to paper4data/{filename_prefix}_*")

def process_results_to_dataframe_sbert(results, sampled_comments=None):
    """Process SBERT results into a DataFrame"""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping DataFrame creation")
        return None, {}
    
    all_hypotheses = set()
    for sector, data in results.items():
        if data and len(data) > 0:
            for batch in data:
                if 'hypotheses' in batch:
                    for hypothesis_name in batch['hypotheses']:
                        all_hypotheses.add(f"{sector}_{hypothesis_name.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}")
    
    rows = []
    for sector, sector_data in results.items():
        if sector_data and len(sector_data) > 0:
            for batch_data in sector_data:
                if 'mean_scores' in batch_data:
                    for comment_text, scores in batch_data['mean_scores'].items():
                        row = {'comment_text': comment_text, 'sector': sector}
                        
                        # Initialize all hypothesis columns
                        for hyp_col in all_hypotheses:
                            row[hyp_col] = 0.0
                        
                        # Set actual scores
                        for hypothesis, score in scores.items():
                            col_name = f"{sector}_{hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}"
                            if col_name in all_hypotheses:
                                row[col_name] = score / 10.0  # Convert to 0-1 scale
                        
                        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("SBERT DataFrame created with shape:", df.shape)
    
    # Create sector labels
    sector_labels = {}
    for sector in ['transport', 'housing', 'food']:
        sector_cols = [col for col in df.columns if col.startswith(f"{sector}_")]
        if sector_cols:
            clean_labels = []
            for col in sector_cols:
                label = col.replace(f"{sector}_", "").replace("_", " ").title()
                clean_labels.append(label)
            sector_labels[sector] = clean_labels
    
    return df.rename(columns={'comment_text': 'comment'}), sector_labels

def main_sbert(comments_by_sector=None, num_runs=10, max_comments_per_sector=100, num_workers=4, batch_size=10, sbert_batch_size=50, threshold=0.5, use_bart=False):
    """
    Main function for SBERT classification (default) or BART classification (optional)
    
    Args:
        comments_by_sector: Dict with sector names as keys and lists of comments as values
        num_runs: Number of classification runs per batch
        max_comments_per_sector: Maximum comments to process per sector
        num_workers: Number of parallel workers
        batch_size: Number of comments per batch
        sbert_batch_size: Internal batch size for SBERT encoding (default: 50)
        threshold: Classification threshold (0.0 to 1.0)
        use_bart: If True, use BART instead of SBERT (default: False)
    """
    if use_bart:
        if not TRANSFORMERS_AVAILABLE:
            print("Error: transformers library not found. Install with: pip install transformers torch")
            return None, None, None
        print("Using BART zero-shot classification")
        return main_bart(comments_by_sector, num_runs, max_comments_per_sector, num_workers, batch_size, threshold)
    else:
        if not SBERT_AVAILABLE:
            print("Error: sentence-transformers library not found. Install with: pip install sentence-transformers")
            return None, None, None
        print("Using SBERT semantic similarity classification")
    
    # Define sector configurations
    sectors_config = {
        'transport': {
            'comments': comments_by_sector.get('transport', []) if comments_by_sector else [],
            'hypotheses': transport_hypotheses
        },
        'housing': {
            'comments': comments_by_sector.get('housing', []) if comments_by_sector else [],
            'hypotheses': housing_hypotheses
        },
        'food': {
            'comments': comments_by_sector.get('food', []) if comments_by_sector else [],
            'hypotheses': food_hypotheses
        }
    }
    
    # Process each sector
    results = {}
    total_start_time = time.time()
    
    for sector, config in sectors_config.items():
        comments = list(config['comments'])[:max_comments_per_sector]
        if len(comments) > 0:
            print(f"\n{'='*60}")
            print(f"Processing {sector} sector with {len(comments)} comments using SBERT")
            print(f"{'='*60}")
            
            sector_start_time = time.time()
            
            results[sector] = process_sector_comments_sbert(
                comments, 
                config['hypotheses'], 
                sector,
                num_runs,
                num_workers,
                batch_size,
                sbert_batch_size
            )
            
            sector_time = time.time() - sector_start_time
            print(f"✓ {sector} completed in {sector_time:.2f} seconds")
        else:
            print(f"No comments found for {sector} sector")
    
    total_time = time.time() - total_start_time
    print(f"\n✓ Total SBERT processing time: {total_time:.2f} seconds")
    
    # Save results
    save_results_sbert(results)
    
    # Process results to DataFrame
    results_df, sector_labels = process_results_to_dataframe_sbert(results, comments_by_sector)
    
    return results, results_df, sector_labels

def main_bart(comments_by_sector=None, num_runs=10, max_comments_per_sector=100, num_workers=4, batch_size=10, threshold=0.5):
    """
    Main function for BART classification (called when use_bart=True)
    
    Args:
        comments_by_sector: Dict with sector names as keys and lists of comments as values
        num_runs: Number of classification runs per batch
        max_comments_per_sector: Maximum comments to process per sector
        num_workers: Number of parallel workers
        batch_size: Number of comments per batch
        threshold: Classification threshold (0.0 to 1.0)
    """
    # Define sector configurations
    sectors_config = {
        'transport': {
            'comments': comments_by_sector.get('transport', []) if comments_by_sector else [],
            'hypotheses': transport_hypotheses
        },
        'housing': {
            'comments': comments_by_sector.get('housing', []) if comments_by_sector else [],
            'hypotheses': housing_hypotheses
        },
        'food': {
            'comments': comments_by_sector.get('food', []) if comments_by_sector else [],
            'hypotheses': food_hypotheses
        }
    }
    
    # Process each sector
    results = {}
    total_start_time = time.time()
    
    for sector, config in sectors_config.items():
        comments = list(config['comments'])[:max_comments_per_sector]
        if len(comments) > 0:
            print(f"\n{'='*60}")
            print(f"Processing {sector} sector with {len(comments)} comments using BART")
            print(f"{'='*60}")
            
            sector_start_time = time.time()
            
            results[sector] = process_sector_comments_bart(
                comments, 
                config['hypotheses'], 
                sector,
                num_runs,
                num_workers,
                batch_size,
                threshold
            )
            
            sector_time = time.time() - sector_start_time
            print(f"✓ {sector} completed in {sector_time:.2f} seconds")
        else:
            print(f"No comments found for {sector} sector")
    
    total_time = time.time() - total_start_time
    print(f"\n✓ Total BART processing time: {total_time:.2f} seconds")
    
    # Save results
    save_results_bart(results)
    
    # Process results to DataFrame
    results_df, sector_labels = process_results_to_dataframe_bart(results, comments_by_sector)
    
    return results, results_df, sector_labels

def save_results_bart(results, filename_prefix='BART_arguments'):
    """Save BART classification results"""
    os.makedirs('paper4data', exist_ok=True)
    
    # Save detailed results
    with open(f'paper4data/{filename_prefix}_detailed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Create summary CSV
    all_scores = []
    all_hypotheses = set()
    
    for sector, sector_results in results.items():
        for batch_result in sector_results:
            if 'mean_scores' in batch_result:
                all_hypotheses.update(batch_result['hypotheses'])
                
                for comment, scores in batch_result['mean_scores'].items():
                    row = {
                        'sector': sector,
                        'comment': comment[:200] + '...' if len(comment) > 200 else comment,
                        'comment_full': comment
                    }
                    
                    for hypothesis in batch_result['hypotheses']:
                        clean_hypothesis = hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
                        score = scores.get(hypothesis, 0.0) / 10.0  # Convert to 0-1 scale
                        row[clean_hypothesis] = score
                    
                    all_scores.append(row)
    
    # Save CSV
    if all_scores:
        import csv
        clean_hypotheses = [h.lower().replace(' ', '_').replace('&', 'and').replace('-', '_') for h in sorted(all_hypotheses)]
        fieldnames = ['sector', 'comment', 'comment_full'] + clean_hypotheses
        
        with open(f'paper4data/{filename_prefix}_scores.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)
    
    # Save JSON summary
    with open(f'paper4data/{filename_prefix}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ BART results saved to paper4data/{filename_prefix}_*")

def process_results_to_dataframe_bart(results, sampled_comments=None):
    """Process BART results into a DataFrame"""
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not available, skipping DataFrame creation")
        return None, {}
    
    all_hypotheses = set()
    for sector, data in results.items():
        if data and len(data) > 0:
            for batch in data:
                if 'hypotheses' in batch:
                    for hypothesis_name in batch['hypotheses']:
                        all_hypotheses.add(f"{sector}_{hypothesis_name.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}")
    
    rows = []
    for sector, sector_data in results.items():
        if sector_data and len(sector_data) > 0:
            for batch_data in sector_data:
                if 'mean_scores' in batch_data:
                    for comment_text, scores in batch_data['mean_scores'].items():
                        row = {'comment_text': comment_text, 'sector': sector}
                        
                        # Initialize all hypothesis columns
                        for hyp_col in all_hypotheses:
                            row[hyp_col] = 0.0
                        
                        # Set actual scores
                        for hypothesis, score in scores.items():
                            col_name = f"{sector}_{hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}"
                            if col_name in all_hypotheses:
                                row[col_name] = score / 10.0  # Convert to 0-1 scale
                        
                        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("BART DataFrame created with shape:", df.shape)
    
    # Create sector labels
    sector_labels = {}
    for sector in ['transport', 'housing', 'food']:
        sector_cols = [col for col in df.columns if col.startswith(f"{sector}_")]
        if sector_cols:
            clean_labels = []
            for col in sector_cols:
                label = col.replace(f"{sector}_", "").replace("_", " ").title()
                clean_labels.append(label)
            sector_labels[sector] = clean_labels
    
    return df.rename(columns={'comment_text': 'comment'}), sector_labels

def test_sbert_small_sample():
    """Test SBERT classification with a small sample"""
    print("=== Testing SBERT Classification with small sample ===")
    
    if not SBERT_AVAILABLE:
        print("Error: sentence-transformers library not available")
        return None
    
    # Create test comments
    test_comments = [
        "I love my Tesla Model 3! The instant torque is amazing and it's so much fun to drive.",
        "The battery replacement cost is really expensive, around $15,000.",
        "I'm worried about range anxiety on long trips.",
        "Solar panels on my roof have really reduced my electricity bills.",
        "The solar farm is an eyesore and ruins the beautiful landscape.",
        "Going vegan has improved my health significantly.",
        "Plant-based meat alternatives are getting cheaper and taste great.",
        "Factory farming is cruel to animals.",
    ]
    
    test_comments_by_sector = {
        'transport': test_comments[:3],
        'housing': test_comments[3:5], 
        'food': test_comments[5:]
    }
    
    # Run SBERT classification
    results, results_df, sector_labels = main_sbert(
        comments_by_sector=test_comments_by_sector,
        num_runs=2,
        max_comments_per_sector=10,
        num_workers=2,
        batch_size=4,
        sbert_batch_size=25,  # Small batch size for testing
        threshold=0.3,
        use_bart=False  # Use SBERT
    )
    
    return results, results_df, sector_labels

def test_bart_small_sample():
    """Test BART classification with a small sample"""
    print("=== Testing BART Classification with small sample ===")
    
    if not TRANSFORMERS_AVAILABLE:
        print("Error: transformers library not available")
        return None
    
    # Create test comments
    test_comments = [
        "I love my Tesla Model 3! The instant torque is amazing and it's so much fun to drive.",
        "The battery replacement cost is really expensive, around $15,000.",
        "I'm worried about range anxiety on long trips.",
        "Solar panels on my roof have really reduced my electricity bills.",
        "The solar farm is an eyesore and ruins the beautiful landscape.",
        "Going vegan has improved my health significantly.",
        "Plant-based meat alternatives are getting cheaper and taste great.",
        "Factory farming is cruel to animals.",
    ]
    
    test_comments_by_sector = {
        'transport': test_comments[:3],
        'housing': test_comments[3:5], 
        'food': test_comments[5:]
    }
    
    # Run BART classification
    results, results_df, sector_labels = main_sbert(
        comments_by_sector=test_comments_by_sector,
        num_runs=2,
        max_comments_per_sector=10,
        num_workers=2,
        batch_size=4,
        threshold=0.3,
        use_bart=True  # Use BART
    )
    
    return results, results_df, sector_labels

if __name__ == "__main__":
    # Test SBERT by default
    test_sbert_small_sample() 