"""
Paper 4 Local LLMs Labelling Classifier
Uses existing survey data to define sector-specific hypotheses (motivators and de-motivators).
Classifies comments into these hypotheses using local LLMs with multiple runs for reliability scoring
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
client = OpenAI(api_key="")
# Sector hypotheses
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
        "Discusses second-life battery uses, recycling logistics, ‘bricking’ or disposal challenges. "
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





# ----------------------------  ENERGY-SURVEY / ATTITUDE TAGS  ---------------------------- #
energy_survey_hypotheses = {
    # prevalence perception
    "Solar_Home_Prevalence": (
        "Perceived commonness of rooftop/home solar panels locally. "
        "Cues: 'extremely common', 'somewhat common', 'on homes'."
    ),
    "Solar_Building_Prevalence": (
        "Perceived commonness of mid-scale solar on buildings/parking lots. "
        "Cues: same scale + 'building', 'parking lot'."
    ),
    "Solar_Farm_Prevalence": (
        "Perceived commonness of large-scale solar farms. "
        "Cues: same scale + 'solar farm', 'utility-scale'."
    ),
    # economic attitudes
    "Renewables_Local_Economy_Help": (
        "Belief a nearby wind/solar project would help the local economy. "
        "Cues: 'help local economy', 'jobs', 'boost'."
    ),
    "Renewables_Local_Economy_Hurt": (
        "Belief it would hurt the local economy. "
        "Cues: 'hurt local economy', 'harm', 'negative impact'."
    ),
    "Renewables_Local_Economy_Neutral_Unsure": (
        "Unsure or no difference on economic impact. "
        "Cues: 'make no difference', 'not sure'."
    ),
    # community impacts
    "Renewables_Landscape_Ugly": (
        "Thinks project would make landscape unattractive. "
        "Cues: 'ugly', 'ruin the view'."
    ),
    "Renewables_Landscape_No": (
        "Believes it would NOT harm landscape aesthetics. "
        "Cues: 'would not make unattractive'."
    ),
    "Renewables_Space_TooMuch": (
        "Concern project takes too much space. "
        "Cues: 'take up too much space', 'footprint'."
    ),
    "Renewables_Space_No": (
        "Believes space taken is acceptable. "
        "Cues: 'would not take too much space'."
    ),
    "Renewables_Utility_Bill_Lower": (
        "Expects local renewables will lower electricity bills. "
        "Cues: 'lower my bill', 'cheaper power'."
    ),
    "Renewables_Utility_Bill_Higher": (
        "Expects it will raise bills. "
        "Cues: 'higher bills', 'more expensive power'."
    ),
    "Renewables_Tax_Revenue_Help": (
        "Believes project will boost local tax revenue. "
        "Cues: 'tax revenue', 'municipal income'."
    ),
    # EV-specific
    "EV_Purchase_Consideration": (
        "Stated likelihood of seriously considering buying an EV. "
        "Cues: 'would very seriously consider', 'somewhat consider', 'down 9 points'."
    ),
    "EV_Env_Comparison": (
        "Views on whether EVs are better/worse for the environment than gas cars. "
        "Cues: 'better for the environment', 'about the same', 'worse'."
    ),
    "EV_Cost_Buy_Upfront": (
        "Beliefs about higher or lower up-front purchase cost. "
        "Cues: 'cost more to buy', 'bigger up-front investment'."
    ),
    "EV_Cost_Charge": (
        "Beliefs about charging cost vs. fueling gas. "
        "Cues: 'cost less to charge', 'cost more to charge'."
    ),
    "EV_Fun_To_Drive": (
        "Perceptions of fun/drive experience vs. gas. "
        "Cues: 'more fun to drive', 'equally fun'."
    ),
    "EV_Reliability_Perception": (
        "Perceived reliability of EVs vs. gas vehicles. "
        "Cues: 'less reliable', 'more reliable'."
    ),
    "EV_Infrastructure_Confidence": (
        "Confidence that the U.S. will build enough charging infrastructure. "
        "Cues: 'extremely confident', 'not at all confident'."
    ),
    # NEW attitude coverage
    "Political_Divide": (
        "Explicit comparison of Democrat vs Republican views on renewables or EVs. "
        "Cues: 'Democrats more likely', 'GOP', 'Republicans think'."
    ),
    "Rural_Skepticism": (
        "Mentions rural residents’ lower support for renewables. "
        "Cues: 'rural Americans', 'country residents', 'farm community skepticism'."
    ),
    "Youth_Optimism": (
        "Highlights younger adults’ greater support for renewables/EVs. "
        "Cues: 'under 30', 'younger adults more likely'."
    ),
}

def load_sector_data():
    """Load sector data from pickle files"""
    try:
        with open('paper4data/sector_keyword_comment_map.pkl', 'rb') as f:
            keyword_comment_map = pickle.load(f)
        return keyword_comment_map
    except FileNotFoundError as e:
        print(f"Error loading sector data: {e}")
        return None

def get_comments_by_sector(keyword_comment_map, sectors):
    """Get comments organized by sector"""
    comments_by_sector = {}
    
    for sector in ['transport', 'housing', 'food']:
        strong_comments = set()
        weak_comments = set()
        
        strong_key = f'{sector}_strong'
        if strong_key in sectors:
            for kw in sectors[strong_key]:
                if kw in keyword_comment_map:
                    strong_comments.update(keyword_comment_map[kw]['matches'])
                
        weak_key = f'{sector}_weak'
        if weak_key in sectors:
            for kw in sectors[weak_key]:
                if kw in keyword_comment_map:
                    weak_comments.update(keyword_comment_map[kw]['matches'])
        
        comments_by_sector[sector] = strong_comments.union(weak_comments)
    
    return comments_by_sector

def create_classification_prompt(hypotheses, sector):
    """Create the classification prompt for GPT"""
    hypothesis_list = "\n".join([f"- {name}: {desc}" for name, desc in hypotheses.items()])
    
    prompt = f"""
You are an expert annotator. Classify Reddit comments about **{sector}** into the hypothesis set below.

{hypothesis_list}

————————————————————
Input format
————————————————————
Comments arrive in one block, each delimited as
[new_comment_id: XXXXXXX]
<comment text>

————————————————————
Labelling rules
————————————————————
1. A comment may trigger **multiple** hypotheses or none.
2. Mark "yes" *only if* the comment contains concrete cue phrases matching the hypothesis:
   - Example: Range-anxiety requires phrases like "range", "miles per charge", "can't make trip"
   - Example: Visual-impact requires phrases like "eyesore", "ugly", "ruin the view", "glare"
3. Generic climate or cost discussion is not a match unless explicitly tied to EV/solar.
4. If one sentence mentions multiple topics, list the ID under every matching hypothesis.
5. Ignore sarcasm or off-hand mentions that do not advance an argument.
6. Do **not** infer or guess intent; rely on surface text.
7. Ignore news-link titles unless the comment itself makes the argument.

————————————————————
Output format (exactly):
{chr(10).join([f"{name}: [comma-separated comment IDs]" for name in hypotheses.keys()])}

• List only IDs, no extra text.
• Omit empty brackets if no IDs match a hypothesis.

Be precise—include a comment ID only when the match is unambiguous.
"""
    
    return prompt

def parse_gpt_response(response, hypotheses):
    """Parse GPT response and extract comment IDs for each hypothesis"""
    if not response:
        return {name: [] for name in hypotheses.keys()}
    
    results = {name: [] for name in hypotheses.keys()}
    
    # Split response into lines
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            hypothesis_name, ids_part = line.split(':', 1)
            hypothesis_name = hypothesis_name.strip()
            
            if hypothesis_name in hypotheses:
                # Extract comment IDs from brackets
                ids_text = ids_part.strip()
                if ids_text.startswith('[') and ids_text.endswith(']'):
                    ids_content = ids_text[1:-1].strip()
                    if ids_content:
                        # Split by comma and clean up
                        comment_ids = [id.strip() for id in ids_content.split(',') if id.strip()]
                        results[hypothesis_name] = comment_ids
    
    return results

def classify_batch_single_run(comments_batch, hypotheses, sector, batch_id, run_id):
    """Classify a batch of comments in a single run"""
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
        prompt = create_classification_prompt(hypotheses, sector)
        full_prompt = prompt + "\n\nComments to classify:\n" + batch_comments
        
        # Call GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert annotator for Reddit comment classification."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        parsed_results = parse_gpt_response(result, hypotheses)
        
        # Convert comment IDs back to actual comments
        final_results = {}
        for hypothesis, comment_ids in parsed_results.items():
            matched_comments = []
            for comment_id in comment_ids:
                # Find the comment by ID
                for comment in comments_batch:
                    if body_to_id[comment] == comment_id:
                        matched_comments.append(comment)
                        break
            final_results[hypothesis] = matched_comments
        
        return {
            'sector': sector,
            'batch_id': batch_id,
            'run_id': run_id,
            'comments': comments_batch,
            'body_to_id': body_to_id,
            'raw_response': result,
            'parsed_results': final_results,
            'hypotheses': list(hypotheses.keys())
        }
        
    except Exception as e:
        print(f"Error in batch {batch_id}, run {run_id}: {str(e)}")
        return None

def classify_batch_multiple_runs(comments_batch, hypotheses, sector, batch_id, num_runs=10):
    """Classify a batch of comments multiple times and calculate mean scores"""
    print(f"Processing {sector} batch {batch_id} with {len(comments_batch)} comments ({num_runs} runs)")
    
    all_runs = []
    
    for run_id in range(1, num_runs + 1):
        result = classify_batch_single_run(comments_batch, hypotheses, sector, batch_id, run_id)
        if result:
            all_runs.append(result)
        
        # Small delay between runs
        time.sleep(0.5)
    
    # Calculate mean scores for each comment and hypothesis
    comment_scores = defaultdict(lambda: defaultdict(list))
    
    for run_result in all_runs:
        for hypothesis, matched_comments in run_result['parsed_results'].items():
            for comment in comments_batch:
                # 1 if comment was matched, 0 if not
                score = 1 if comment in matched_comments else 0
                comment_scores[comment][hypothesis].append(score)
    
    # Calculate mean scores (0/10 to 10/10)
    mean_scores = {}
    for comment in comments_batch:
        comment_mean_scores = {}
        for hypothesis in hypotheses.keys():
            scores = comment_scores[comment][hypothesis]
            if scores:
                mean_score = sum(scores) / len(scores)  # This will be 0.0 to 1.0
                # Convert to 0/10 to 10/10 scale
                mean_score_10 = mean_score * 10
                comment_mean_scores[hypothesis] = mean_score_10
            else:
                comment_mean_scores[hypothesis] = 0.0
        mean_scores[comment] = comment_mean_scores
    
    return {
        'sector': sector,
        'batch_id': batch_id,
        'num_runs': num_runs,
        'all_runs': all_runs,
        'mean_scores': mean_scores,
        'hypotheses': list(hypotheses.keys())
    }

def process_sector_batch(client, assistant_id, comments_batch, hypotheses, sector, batch_id, shared_progress):
    """Process a batch of comments for hypothesis classification"""
    print(f"Processing {sector} batch {batch_id} with {len(comments_batch)} comments")
    
    # Create mapping from comment to its ID within this batch
    body_to_id = {comment: f'comment{i+1}' for i, comment in enumerate(comments_batch)}
    
    # Create batch string with comments
    batch_comments = ''
    for comment in comments_batch:
        comment_id = body_to_id[comment]
        comment_str = f'\n[new_comment_id: {comment_id}]: {comment}'
        batch_comments += comment_str
    
    # Create prompt
    prompt = create_classification_prompt(hypotheses, sector)
    full_prompt = prompt + "\n\nComments to classify:\n" + batch_comments
    
    # Call GPT
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert annotator for Reddit comment classification."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.1,
        max_tokens=2000
    )
    
    result = response.choices[0].message.content.strip()
    
    # Parse the response
    parsed_results = parse_gpt_response(result, hypotheses)
    
    # Convert comment IDs back to actual comments
    final_results = {}
    for hypothesis, comment_ids in parsed_results.items():
        matched_comments = []
        for comment_id in comment_ids:
            # Find the comment by ID
            for comment in comments_batch:
                if body_to_id[comment] == comment_id:
                    matched_comments.append(comment)
                    break
        final_results[hypothesis] = matched_comments
    
    # Update progress
    with shared_progress['lock']:
        shared_progress['processed'] += len(comments_batch)
        shared_progress['pbar'].update(len(comments_batch))
    
    # Save batch result with metadata
    batch_result = {
        'sector': sector,
        'batch_id': batch_id,
        'comments': comments_batch,
        'body_to_id': body_to_id,
        'raw_response': result,
        'parsed_results': final_results,
        'hypotheses': list(hypotheses.keys())
    }
    
    return batch_result

def process_sector_comments(comments, hypotheses, sector, num_runs=10, num_agents=20, batch_size=10):
    """Process all comments for a sector using fully parallel multiple runs for reliability scoring"""
    
    # Sample comments if too many
    sampled_comments = list(comments)[:100] if len(comments) > 100 else list(comments)
    
    if len(sampled_comments) == 0:
        return []
    
    print(f"Processing {len(sampled_comments)} comments with {num_agents} parallel agents, batch size {batch_size}, {num_runs} runs per batch")
    
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
                'sector': sector
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
                classify_batch_single_run,
                task['comments'],
                task['hypotheses'],
                task['sector'],
                task['batch_id'],
                task['run_id']
            )
            future_to_task[future] = task
        
        # Use a single progress bar for all tasks
        with tqdm(total=total_tasks, desc=f"Processing {sector} batches", unit="task") as pbar:
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
        
        # Calculate mean scores for each comment and hypothesis
        comment_scores = defaultdict(lambda: defaultdict(list))
        
        for run_result in batch_runs:
            for hypothesis, matched_comments in run_result['parsed_results'].items():
                for comment in comments_batch:
                    # 1 if comment was matched, 0 if not
                    score = 1 if comment in matched_comments else 0
                    comment_scores[comment][hypothesis].append(score)
        
        # Calculate mean scores (0/10 to 10/10)
        mean_scores = {}
        for comment in comments_batch:
            comment_mean_scores = {}
            for hypothesis in hypotheses.keys():
                scores = comment_scores[comment][hypothesis]
                if scores:
                    mean_score = sum(scores) / len(scores)  # This will be 0.0 to 1.0
                    # Convert to 0/10 to 10/10 scale
                    mean_score_10 = mean_score * 10
                    comment_mean_scores[hypothesis] = mean_score_10
                else:
                    comment_mean_scores[hypothesis] = 0.0
            mean_scores[comment] = comment_mean_scores
        
        # Create consolidated batch result
        batch_result = {
            'sector': sector,
            'batch_id': batch_id,
            'num_runs': len(batch_runs),
            'all_runs': batch_runs,
            'mean_scores': mean_scores,
            'hypotheses': list(hypotheses.keys()),
            'comments': comments_batch  # Add this for compatibility with save_results
        }
        
        batch_results.append(batch_result)
    
    print(f"Processed {len(batch_results)} batches with {len(all_run_results)} successful runs")
    
    return batch_results

def save_results(results, filename_prefix='GPT_arguments'):
    """Save results to files"""
    os.makedirs('paper4data', exist_ok=True)
    
    # Save detailed results
    with open(f'paper4data/{filename_prefix}_detailed.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Collect all unique hypotheses across all batches
    all_hypotheses = set()
    for sector, sector_results in results.items():
        for batch_result in sector_results:
            all_hypotheses.update(batch_result['hypotheses'])
    
    # Create summary DataFrame
    all_scores = []
    for sector, sector_results in results.items():
        for batch_result in sector_results:
            # Use mean_scores instead of comments and parsed_results
            for comment, scores in batch_result['mean_scores'].items():
                row = {
                    'sector': sector,
                    'batch_id': batch_result['batch_id'],
                    'comment': comment[:200] + '...' if len(comment) > 200 else comment,
                    'comment_full': comment
                }
                # Use the fractional scores from mean_scores
                for hypothesis in all_hypotheses:
                    # Use clean fieldname for CSV
                    clean_hypothesis = hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')
                    # Get the score for this hypothesis (0.0 to 1.0), default to 0.0 if not present
                    score = scores.get(hypothesis, 0.0) / 10.0  # Convert from 0-10 scale to 0-1 scale
                    row[clean_hypothesis] = score
                all_scores.append(row)
    
    if all_scores:
        # Create clean fieldnames for CSV from all hypotheses
        clean_hypotheses = [h.lower().replace(' ', '_').replace('&', 'and').replace('-', '_') for h in sorted(all_hypotheses)]
        fieldnames = ['sector', 'batch_id', 'comment', 'comment_full'] + clean_hypotheses
        
        # Write CSV with UTF-8 encoding to handle Unicode characters
        with open(f'paper4data/{filename_prefix}_scores.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)
        
        # Save summary statistics
        summary_stats = {}
        for sector in results.keys():
            sector_scores = []
            for batch_result in results[sector]:
                for comment, scores in batch_result['mean_scores'].items():
                    comment_scores = {}
                    for hypothesis in batch_result['hypotheses']:
                        # Convert from 0-10 scale to 0-1 scale for summary
                        score = scores.get(hypothesis, 0.0) / 10.0
                        comment_scores[hypothesis] = score
                    sector_scores.append(comment_scores)
            
            if sector_scores:
                # Calculate mean scores manually
                hypothesis_means = {}
                hypothesis_names = list(sector_scores[0].keys())
                for hypothesis in hypothesis_names:
                    values = [score[hypothesis] for score in sector_scores if hypothesis in score]
                    if values:
                        hypothesis_means[hypothesis] = sum(values) / len(values)
                    else:
                        hypothesis_means[hypothesis] = 0.0
                
                summary_stats[sector] = {
                    'total_comments': len(sector_scores),
                    'mean_scores_by_hypothesis': hypothesis_means
                }
        
        # Write JSON with UTF-8 encoding to handle Unicode characters
        with open(f'paper4data/{filename_prefix}_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to paper4data/{filename_prefix}_*")

def process_results_to_dataframe(results, sampled_comments=None):
    """Process results into a DataFrame with proper formatting"""
    import pandas as pd
    
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
                # Use mean_scores data structure
                if 'mean_scores' in batch_data:
                    for comment_text, scores in batch_data['mean_scores'].items():
                        row = {'comment_text': comment_text, 'sector': sector}
                        
                        # Initialize all hypothesis columns to 0.0
                        for hyp_col in all_hypotheses:
                            row[hyp_col] = 0.0
                        
                        # Set actual scores for this comment
                        for hypothesis, score in scores.items():
                            col_name = f"{sector}_{hypothesis.lower().replace(' ', '_').replace('&', 'and').replace('-', '_')}"
                            if col_name in all_hypotheses:
                                # Convert from 0-10 scale to 0-1 scale
                                row[col_name] = score / 10.0
                        
                        rows.append(row)
    
    # Add missing comments from sampled_comments if provided
    if sampled_comments:
        for sector, comments in sampled_comments.items():
            existing_comments = set([r['comment_text'] for r in rows if r['sector'] == sector])
            missing_comments = [c for c in comments if c not in existing_comments]
            
            for comment in missing_comments:
                row = {'comment_text': comment, 'sector': sector}
                for hyp_col in all_hypotheses:
                    row[hyp_col] = 0.0
                rows.append(row)
    
    # Create final DataFrame with all comments
    df = pd.DataFrame(rows)
    print("DataFrame created with shape:", df.shape)
    print(f"Comments per sector:")
    for sector in df['sector'].unique():
        count = len(df[df['sector'] == sector])
        print(f"- {sector}: {count} comments")
    
    results_df = df.copy()
    for col in results_df.columns:
        if col not in ['comment_text', 'sector']:
            results_df[col] = results_df[col].astype(float)
    results_df = results_df.rename(columns={'comment_text': 'comment'})
    
    sector_labels = {}
    for sector in ['transport', 'housing', 'food']:
        sector_cols = [col for col in results_df.columns if col.startswith(f"{sector}_")]
        if sector_cols:
            clean_labels = []
            for col in sector_cols:
                label = col.replace(f"{sector}_", "").replace("_", " ").title()
                clean_labels.append(label)
            sector_labels[sector] = clean_labels
    
    print("\nSector labels for plotting:")
    print(sector_labels)
    
    print("\nSample scores (should be between 0.0 and 1.0):")
    for sector in ['transport', 'housing', 'food']:
        sector_cols = [col for col in results_df.columns if col.startswith(f"{sector}_")]
        if sector_cols:
            print(f"{sector}: {results_df[sector_cols[:2]].describe()}")
    
    return results_df, sector_labels

def main(comments_by_sector=None, num_runs=10, max_comments_per_sector=100, num_agents=20, batch_size=10, use_survey_frames=False):
    """Main function to process all sectors
    
    Args:
        comments_by_sector: Dict with sector names as keys and lists of comments as values
                           Format: {'transport': [comment1, comment2, ...], 
                                   'housing': [comment1, comment2, ...], 
                                   'food': [comment1, comment2, ...]}
        num_runs: Number of classification runs per batch (default: 10)
        max_comments_per_sector: Maximum comments to process per sector (default: 100)
        num_agents: Number of parallel agents to use (default: 20)
        batch_size: Number of comments per batch (default: 10)
        use_survey_frames: If True, use energy_survey_hypotheses instead of sector-specific hypotheses (default: False)
    
    Returns:
        tuple: (results, results_df, sector_labels)
    """
    
    if use_survey_frames:
        # When using survey frames, treat all comments as one unified dataset
        print("Using energy survey hypotheses instead of sector-specific hypotheses")
        
        # Combine all comments from all sectors
        all_comments = []
        if comments_by_sector:
            for sector, sector_comments in comments_by_sector.items():
                all_comments.extend(list(sector_comments)[:max_comments_per_sector])
        
        if len(all_comments) > 0:
            print(f"\nProcessing {len(all_comments)} comments with survey frames...")
            start_time = time.time()
            
            # Process all comments with survey hypotheses
            survey_results = process_sector_comments(
                all_comments, 
                energy_survey_hypotheses, 
                'energy_survey',
                num_runs,
                num_agents,
                batch_size
            )
            
            end_time = time.time()
            print(f"Survey frame processing completed in {end_time - start_time:.2f} seconds")
            
            # Package results
            results = {'energy_survey': survey_results}
            
        else:
            print("No comments found to process with survey frames.")
            results = {}
        
    else:
        # Original sector-specific processing
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
                print(f"\nProcessing {sector} sector with {len(comments)} comments...")
                sector_start_time = time.time()
                
                results[sector] = process_sector_comments(
                    comments, 
                    config['hypotheses'], 
                    sector,
                    num_runs,
                    num_agents,
                    batch_size
                )
                
                sector_time = time.time() - sector_start_time
                print(f"{sector} completed in {sector_time:.2f} seconds")
            else:
                print(f"No comments found for {sector} sector")
        
        total_time = time.time() - total_start_time
        print(f"\nTotal processing time: {total_time:.2f} seconds")
    
    # Save results
    save_results(results)
    
    # Process results to DataFrame
    results_df, sector_labels = process_results_to_dataframe(results, comments_by_sector)
    
    return results, results_df, sector_labels

def main_from_files(sectors=None, num_runs=10, max_comments_per_sector=100, num_agents=20, batch_size=10):
    """Main function to process all sectors by loading data from files (legacy function)"""
    print("Loading sector data...")
    keyword_comment_map = load_sector_data()
    
    if keyword_comment_map is None:
        print("Failed to load sector data. Exiting.")
        return None
    
    print(f"Loaded data with {len(keyword_comment_map)} keywords")
    
    # Get comments by sector
    if sectors is None:
        # Load sectors from the keyword list
        try:
            with open('paper4data/sector_keyword_list.pkl', 'rb') as f:
                sectors = pickle.load(f)
        except:
            print("Could not load sectors, using default structure")
            sectors = {
                'transport_strong': ['ev', 'electric vehicle'],
                'transport_weak': ['tesla'],
                'housing_strong': ['solar', 'rooftop solar'],
                'housing_weak': ['solar panel'],
                'food_strong': ['vegan', 'plant-based'],
                'food_weak': ['meat']
            }
    
    comments_by_sector = get_comments_by_sector(keyword_comment_map, sectors)
    
    # Convert sets to lists for consistency
    comments_by_sector = {sector: list(comments) for sector, comments in comments_by_sector.items()}
    
    return main(comments_by_sector, num_runs, max_comments_per_sector, num_agents, batch_size)

def process_single_sector(comments, sector, num_runs=10, max_comments=None, num_agents=20, batch_size=10, use_survey_frames=False):
    """Process a single sector with custom comments
    
    Args:
        comments: List of comments to classify
        sector: Sector name ('transport', 'housing', 'food', or 'energy_survey' if use_survey_frames=True)
        num_runs: Number of classification runs per batch (default: 10)
        max_comments: Maximum comments to process (default: None, process all)
        num_agents: Number of parallel agents to use (default: 20)
        batch_size: Number of comments per batch (default: 10)
        use_survey_frames: If True, use energy_survey_hypotheses instead of sector-specific hypotheses (default: False)
    
    Returns:
        tuple: (sector_results, results_df, sector_labels)
    """
    # Limit comments if specified
    if max_comments:
        comments = comments[:max_comments]
    
    if use_survey_frames:
        # Use survey hypotheses
        sector_hypotheses = energy_survey_hypotheses
        sector = 'energy_survey'  # Override sector name for survey frames
        print(f"Using energy survey hypotheses for {len(comments)} comments...")
    else:
        # Get hypotheses for the sector
        sector_hypotheses = {
            'transport': transport_hypotheses,
            'housing': housing_hypotheses,
            'food': food_hypotheses
        }.get(sector)
        
        if not sector_hypotheses:
            print(f"Unknown sector: {sector}. Must be 'transport', 'housing', or 'food'.")
            return None, None, None
        
        print(f"Processing {sector} sector with {len(comments)} comments...")
    
    # Process the comments
    results = process_sector_comments(comments, sector_hypotheses, sector, num_runs, num_agents, batch_size)
    
    # Save results for this sector
    sector_results = {sector: results}
    save_results(sector_results, filename_prefix=f'GPT_arguments_{sector}')
    
    # Process results to DataFrame
    results_df, sector_labels = process_results_to_dataframe(sector_results, {sector: comments})
    
    return sector_results, results_df, sector_labels

def test_small_sample():
    """Test with a small sample of 10 comments"""
    print("=== Testing with small sample (10 comments) ===")
    
    # Create a small test sample
    test_comments = [
        "I love my Tesla Model 3! The instant torque is amazing and it's so much fun to drive.",
        "The battery replacement cost is really expensive, around $15,000.",
        "I'm worried about range anxiety on long trips.",
        "Solar panels on my roof have really reduced my electricity bills.",
        "The solar farm is an eyesore and ruins the beautiful landscape.",
        "Going vegan has improved my health significantly.",
        "Plant-based meat alternatives are getting cheaper and taste great.",
        "Factory farming is cruel to animals.",
        "The charging infrastructure needs to be expanded.",
        "Electric vehicles have zero tailpipe emissions."
    ]
    
    print("\n--- Testing Regular Sector Hypotheses ---")
    # Test with transport sector (regular hypotheses)
    sector_results, sector_df, sector_labels = process_single_sector(
        comments=test_comments,
        sector='transport',
        num_runs=3,
        num_agents=5,
        batch_size=5,
        use_survey_frames=False
    )
    
    print("\n--- Testing Survey Frame Hypotheses ---")
    # Test with survey frames
    survey_results, survey_df, survey_labels = process_single_sector(
        comments=test_comments,
        sector='energy_survey',  # This will be overridden anyway
        num_runs=3,
        num_agents=5,
        batch_size=5,
        use_survey_frames=True
    )
    
    return sector_results, survey_results

if __name__ == "__main__":
    # Test with small sample
    test_small_sample() 