"""
vLLM Server Test - Classifier for Survey Questions
Standalone script that connects to vLLM server for survey question classification.
vLLM server runs on localhost:8000-8003 (multiple models on different ports)

Optimized for maximum throughput using async requests and continuous batching.
vLLM automatically batches concurrent requests together for efficient processing.

This script is standalone and includes all necessary functions and data.
"""

import json
import requests
import aiohttp
import asyncio
from typing import List, Dict, Tuple
import time
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import jsonschema
from jsonschema import validate, ValidationError
import concurrent.futures
from functools import partial
from collections import Counter
import sys
import os
import pandas as pd
import random

# Load API configuration (for endpoint structure)
with open('paper4_LOCAL_LLMS_api.json', 'r') as f:
    api_config = json.load(f)

# ============================================================================
# Survey Questions and Sample Data (standalone - no dependency on sample_local_llm.py)
# ============================================================================

def get_default_questions() -> Dict:
    """Return default hardcoded survey questions"""
    return {
    "Local_Economy_Help": {
        "question": "Would installing a wind or solar power development in your community help your local economy?",
        "description": "Belief that nearby wind/solar project would help local economy. Cues: 'help local economy', 'jobs', 'boost', 'economic growth'."
    },
    "Local_Economy_Hurt": {
        "question": "Would installing a wind or solar power development in your community hurt your local economy?",
        "description": "Belief that nearby wind/solar project would hurt local economy. Cues: 'hurt local economy', 'harm', 'negative impact', 'economic decline'."
    },
    "Local_Economy_No_Difference": {
        "question": "Would installing a wind or solar power development in your community make no difference to your local economy?",
        "description": "Belief that nearby wind/solar project would make no difference to local economy. Cues: 'make no difference', 'no impact', 'neutral'."
    },
    "Landscape_Unattractive": {
        "question": "Would installing a wind or solar power development in your community make the landscape unattractive?",
        "description": "Belief that project would make landscape unattractive. Cues: 'ugly', 'ruin the view', 'eyesore', 'unattractive', 'aesthetic harm'."
    },
    "Landscape_Not_Unattractive": {
        "question": "Would installing a wind or solar power development in your community NOT make the landscape unattractive?",
        "description": "Belief that project would NOT harm landscape aesthetics. Cues: 'would not make unattractive', 'beautiful', 'aesthetic benefit'."
    },
    "Space_Too_Much": {
        "question": "Would installing a wind or solar power development in your community take up too much space?",
        "description": "Concern that project takes too much space. Cues: 'take up too much space', 'footprint', 'land use', 'too large'."
    },
    "Space_Acceptable": {
        "question": "Would installing a wind or solar power development in your community NOT take up too much space?",
        "description": "Belief that space taken is acceptable. Cues: 'would not take too much space', 'reasonable footprint', 'acceptable size'."
    },
    "Utility_Bill_Lower": {
        "question": "Would installing a wind or solar power development in your community lower the price you pay for electricity?",
        "description": "Expectation that local renewables will lower electricity bills. Cues: 'lower my bill', 'cheaper power', 'reduce costs', 'savings'."
    },
    "Utility_Bill_Higher": {
        "question": "Would installing a wind or solar power development in your community raise the price you pay for electricity?",
        "description": "Expectation that local renewables will raise electricity bills. Cues: 'higher bills', 'more expensive power', 'cost increase'."
    },
    "Tax_Revenue_Help": {
        "question": "Would installing a wind or solar power development in your community help local tax revenue?",
        "description": "Belief that project will boost local tax revenue. Cues: 'tax revenue', 'municipal income', 'local taxes', 'revenue boost'."
    }
}

def load_survey_questions(use_real_survey_questions: bool = False) -> Dict:
    """Load survey questions from JSON file or use default hardcoded questions"""
    if use_real_survey_questions:
        # Load from survey_question.json
        if not os.path.exists('survey_question.json'):
            print("Error: survey_question.json not found. Using default questions.")
            return get_default_questions()
        
        with open('survey_question.json', 'r', encoding='utf-8') as f:
            survey_data = json.load(f)
        
        # Flatten the nested structure (Food, Housing, Energy) into a single dict
        flattened = {}
        for sector, questions in survey_data.items():
            for question_id, question_info in questions.items():
                # Keep the question_id as-is, or prefix with sector if needed
                flattened[question_id] = question_info
        
        print(f"Loaded {len(flattened)} questions from survey_question.json")
        print(f"Sectors: {', '.join(survey_data.keys())}")
        return flattened
    else:
        return get_default_questions()

# Generate sample statements about wind and solar power (including vague and irrelevant ones)
SAMPLE_STATEMENTS = [
    # Clear, relevant statements
    "Installing a solar farm in our town would create hundreds of local jobs and boost our economy.",
    "Those wind turbines are an eyesore and completely ruin the beautiful countryside view.",
    "I'm not sure if a solar panel farm would really help or hurt our local economy - it's hard to say.",
    "The solar development takes up way too much farmland that we need for agriculture.",
    "Having renewable energy nearby would definitely lower my electricity bills, which is great.",
    "I'm worried that building a wind farm will increase our electricity costs because of infrastructure expenses.",
    "The new solar panels on the community center look great and don't detract from the area at all.",
    "A wind farm would bring in significant tax revenue for our county, helping fund schools and roads.",
    "I don't think a solar development would make any real difference to our local economy one way or another.",
    "The space used for the solar farm is reasonable compared to the benefits we'll get from clean energy.",
    # More clear examples
    "Solar panels on rooftops are becoming more common in our neighborhood.",
    "Wind energy is clean and doesn't pollute the air like coal plants do.",
    "The construction of the wind farm disrupted local wildlife habitats significantly.",
    "Our electricity rates went down after the solar project started operating.",
    "The solar installation created about 50 construction jobs for local workers.",
    # Vague/ambiguous statements
    "Renewable energy is interesting, I guess.",
    "Some people like solar, others don't really care about it.",
    "Energy projects can have various impacts on communities.",
    "It depends on how you look at it, really.",
    "There are pros and cons to everything.",
    # Irrelevant statements (not about wind/solar development)
    "I went to the grocery store yesterday and bought some apples.",
    "The weather has been really nice this week.",
    "My favorite TV show is on tonight.",
    "Traffic was terrible on the highway this morning.",
    "I'm thinking about renovating my kitchen next year.",
    # Borderline relevant (tangentially related)
    "Climate change is a serious issue that needs addressing.",
    "Electric cars are becoming more popular these days.",
    "I read an article about energy policy in the newspaper.",
    "The government should invest more in infrastructure.",
    "Technology is advancing rapidly in many fields."
]

# Ground truth labels: {statement_index: {question_id: 'yes'/'no'}}
GROUND_TRUTH = {
    0: {  # "Installing a solar farm in our town would create hundreds of local jobs and boost our economy."
        "Local_Economy_Help": "yes",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    1: {  # "Those wind turbines are an eyesore and completely ruin the beautiful countryside view."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "yes",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    2: {  # "I'm not sure if a solar panel farm would really help or hurt our local economy - it's hard to say."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "yes",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    3: {  # "The solar development takes up way too much farmland that we need for agriculture."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "yes",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    4: {  # "Having renewable energy nearby would definitely lower my electricity bills, which is great."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "yes",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    5: {  # "I'm worried that building a wind farm will increase our electricity costs because of infrastructure expenses."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "yes",
        "Tax_Revenue_Help": "no"
    },
    6: {  # "The new solar panels on the community center look great and don't detract from the area at all."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "yes",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    7: {  # "A wind farm would bring in significant tax revenue for our county, helping fund schools and roads."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "yes"
    },
    8: {  # "I don't think a solar development would make any real difference to our local economy one way or another."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "yes",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    9: {  # "The space used for the solar farm is reasonable compared to the benefits we'll get from clean energy."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "yes",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    10: {  # "Solar panels on rooftops are becoming more common in our neighborhood."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "yes",
        "Space_Too_Much": "no",
        "Space_Acceptable": "yes",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    11: {  # "Wind energy is clean and doesn't pollute the air like coal plants do."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    12: {  # "The construction of the wind farm disrupted local wildlife habitats significantly."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "yes",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "yes",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    13: {  # "Our electricity rates went down after the solar project started operating."
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "yes",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    14: {  # "The solar installation created about 50 construction jobs for local workers."
        "Local_Economy_Help": "yes",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    15: {  # "Renewable energy is interesting, I guess." - Vague
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    16: {  # "Some people like solar, others don't really care about it." - Vague
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    17: {  # "Energy projects can have various impacts on communities." - Vague
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    18: {  # "It depends on how you look at it, really." - Vague
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    19: {  # "There are pros and cons to everything." - Vague
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    20: {  # "I went to the grocery store yesterday and bought some apples." - Irrelevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    21: {  # "The weather has been really nice this week." - Irrelevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    22: {  # "My favorite TV show is on tonight." - Irrelevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    23: {  # "Traffic was terrible on the highway this morning." - Irrelevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    24: {  # "I'm thinking about renovating my kitchen next year." - Irrelevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    25: {  # "Climate change is a serious issue that needs addressing." - Borderline relevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    26: {  # "Electric cars are becoming more popular these days." - Borderline relevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    27: {  # "I read an article about energy policy in the newspaper." - Borderline relevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    28: {  # "The government should invest more in infrastructure." - Borderline relevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    },
    29: {  # "Technology is advancing rapidly in many fields." - Borderline relevant
        "Local_Economy_Help": "no",
        "Local_Economy_Hurt": "no",
        "Local_Economy_No_Difference": "no",
        "Landscape_Unattractive": "no",
        "Landscape_Not_Unattractive": "no",
        "Space_Too_Much": "no",
        "Space_Acceptable": "no",
        "Utility_Bill_Lower": "no",
        "Utility_Bill_Higher": "no",
        "Tax_Revenue_Help": "no"
    }
}

# JSON Schema for API response validation
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant": {
            "type": "string",
            "enum": ["yes", "no"]
        }
    },
    "required": ["relevant"],
    "additionalProperties": False
}

def group_related_questions(questions: Dict) -> Dict[str, List[Tuple[str, Dict]]]:
    """
    Group related questions by their base topic.
    Questions like "Local_Economy_Help" and "Local_Economy_Hurt" are grouped together
    since they share the same base topic "Local_Economy".
    
    Handles special cases like:
    - "Landscape_Unattractive" and "Landscape_Not_Unattractive" -> both grouped as "Landscape"
    - "Space_Too_Much" and "Space_Acceptable" -> both grouped as "Space"
    
    Returns:
        Dictionary mapping base_topic -> list of (question_id, question_info) tuples
    """
    # Define variant suffixes that should be stripped to find the base topic
    variant_suffixes = [
        '_Help', '_Hurt', '_No_Difference',
        '_Lower', '_Higher',
        '_Unattractive', '_Not_Unattractive',
        '_Too_Much', '_Acceptable'
    ]
    
    groups = {}
    for question_id, question_info in questions.items():
        # Try to find base topic by removing known variant suffixes
        base_topic = question_id
        for suffix in variant_suffixes:
            if question_id.endswith(suffix):
                base_topic = question_id[:-len(suffix)]
                break
        
        # If no variant suffix matched, try splitting on last underscore as fallback
        if base_topic == question_id:
            parts = question_id.split('_')
            if len(parts) > 1:
                base_topic = '_'.join(parts[:-1])
        
        if base_topic not in groups:
            groups[base_topic] = []
        groups[base_topic].append((question_id, question_info))
    
    return groups

def create_relevance_prompt(statement: str, question_info: Dict) -> str:
    """Create prompt for first pass: determine if statement is relevant to question"""
    prompt = f"""Please determine if this statement is relevant to the survey question below.

Survey Question: {question_info['question']}

Statement: "{statement}"

Is this statement relevant to the survey question? Does it express an opinion, belief, or concern related to this question?

Please respond with valid JSON:
{{
  "relevant": "yes"  // or "no"
}}

- "yes" if the statement is relevant to this survey question
- "no" if the statement is not relevant to this survey question

Response (JSON only):"""
    return prompt

def create_relevance_prompt_grouped(statement: str, grouped_questions: List[Tuple[str, Dict]]) -> str:
    """
    Create prompt for first pass: determine if statement is relevant to a group of related questions.
    This is more efficient than checking each question separately.
    
    Args:
        statement: The statement to check
        grouped_questions: List of (question_id, question_info) tuples for related questions
    
    Returns:
        Prompt string
    """
    questions_text = "\n".join([
        f"- {q_info['question']}" for _, q_info in grouped_questions
    ])
    
    prompt = f"""Please determine if this statement is relevant to ANY of the following related survey questions.

Related Survey Questions:
{questions_text}

Statement: "{statement}"

Is this statement relevant to any of these survey questions? Does it express an opinion, belief, or concern related to any of these questions?

Please respond with valid JSON:
{{
  "relevant": "yes"  // or "no"
}}

- "yes" if the statement is relevant to any of these survey questions
- "no" if the statement is not relevant to any of these survey questions

Response (JSON only):"""
    return prompt

def create_classification_prompt(statement: str, question_info: Dict) -> str:
    """Create prompt for second pass: detailed classification (only for relevant statements)"""
    prompt = f"""Please classify this statement about wind and solar power development.

Survey Question: {question_info['question']}

Question Description: {question_info['description']}

Statement to classify: "{statement}"

This statement has been identified as relevant to the survey question above. Now determine: Does this statement express support for, agreement with, or alignment with this survey question?

Please respond with valid JSON in the following format:
{{
  "relevant": "yes"  // or "no"
}}

- Please use "yes" if the statement supports, agrees with, or aligns with this survey question
- Please use "no" if the statement does not support, contradicts, or is not aligned with this survey question

Response (JSON only):"""
    return prompt

def check_relevance(statement: str, question_id: str, question_info: Dict, 
                    model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Tuple[str, str]:
    """
    First pass: Check if statement is relevant to question
    
    Returns:
        Tuple of (question_id, response) where response is 'yes', 'no', or 'error'
    """
    prompt = create_relevance_prompt(statement, question_info)
    # Note: This function is kept for compatibility but not used in vLLM version
    # vLLM uses async versions instead
    return (question_id, 'error')

def classify_statement_against_question(statement: str, question_id: str, question_info: Dict, 
                                        model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Tuple[str, str]:
    """
    Second pass: Classify a relevant statement against a survey question
    
    Returns:
        Tuple of (question_id, response) where response is 'yes', 'no', or 'error'
    """
    prompt = create_classification_prompt(statement, question_info)
    # Note: This function is kept for compatibility but not used in vLLM version
    # vLLM uses async versions instead
    return (question_id, 'error')

def check_relevance_grouped(statement: str, grouped_questions: List[Tuple[str, Dict]], 
                            model_config: Dict, base_url: str = "http://127.0.0.1:1234") -> Dict[str, str]:
    """
    First pass: Check if statement is relevant to a group of related questions.
    Returns the same relevance result for all questions in the group.
    
    Returns:
        Dictionary mapping question_id -> 'yes'/'no'/'error'
    """
    prompt = create_relevance_prompt_grouped(statement, grouped_questions)
    # Note: This function is kept for compatibility but not used in vLLM version
    # vLLM uses async versions instead
    return {question_id: 'error' for question_id, _ in grouped_questions}

def check_relevance_task(args: Tuple) -> Tuple[int, str, str]:
    """First pass: Check relevance of a single statement-question pair"""
    # Note: This function is kept for compatibility but not used in vLLM version
    stmt_idx, statement, question_id, question_info, model_config, base_url = args
    return (stmt_idx, question_id, 'error')

def classify_single_task(args: Tuple) -> Tuple[int, str, str]:
    """Second pass: Classify a relevant statement-question pair"""
    # Note: This function is kept for compatibility but not used in vLLM version
    stmt_idx, statement, question_id, question_info, model_config, base_url = args
    return (stmt_idx, question_id, 'error')

def first_pass_relevance(statements: List[str], questions: Dict, 
                        model_config: Dict, base_url: str = "http://127.0.0.1:1234",
                        max_workers: int = 50) -> Dict:
    """
    First pass: Determine relevance of all statements to all questions
    Uses grouped questions to reduce API calls (e.g., Help/Hurt variants checked together)
    
    Note: This function is kept for compatibility but not used in vLLM version.
    vLLM uses async versions instead.
    
    Returns:
        Dictionary with structure: {statement_index: {question_id: 'yes'/'no'/'error'}}
    """
    # Return empty results - vLLM version uses async functions
    results = {}
    for stmt_idx in range(len(statements)):
        results[stmt_idx] = {}
        for question_id in questions.keys():
            results[stmt_idx][question_id] = 'error'
    return results

def second_pass_classification(statements: List[str], questions: Dict, relevance_results: Dict,
                              model_config: Dict, base_url: str = "http://127.0.0.1:1234",
                              max_workers: int = 50) -> Dict:
    """
    Second pass: Detailed classification only for statements marked as relevant
    
    Note: This function is kept for compatibility but not used in vLLM version.
    vLLM uses async versions instead.
    
    Returns:
        Dictionary with structure: {statement_index: {question_id: 'yes'/'no'/'error'}}
    """
    # Return empty results - vLLM version uses async functions
    results = {}
    for stmt_idx in range(len(statements)):
        results[stmt_idx] = {}
        for question_id in questions.keys():
            results[stmt_idx][question_id] = 'error'
    return results

def calculate_coherence(responses: List[str]) -> float:
    """Calculate coherence as percentage of majority label / total models"""
    if not responses:
        return 0.0
    
    # Count occurrences of each response (excluding errors)
    valid_responses = [r for r in responses if r != 'error']
    if not valid_responses:
        return 0.0
    
    counts = Counter(valid_responses)
    majority_count = max(counts.values())
    total_models = len(responses)
    
    # Coherence = majority count / total models
    coherence = (majority_count / total_models) * 100.0
    return round(coherence, 2)

def get_majority_vote(responses: List[str]) -> str:
    """Get majority vote from responses (excluding errors)"""
    valid_responses = [r for r in responses if r != 'error']
    if not valid_responses:
        return 'error'
    counts = Counter(valid_responses)
    return counts.most_common(1)[0][0]

def save_results_csv(all_results: Dict[str, Dict], statements: List[str], questions: Dict, filename: str = "sample_local_llm_results.csv", has_ground_truth: bool = True, statement_to_original_index: Dict[int, int] = None):
    """
    Save results to CSV file with statement text, model classifications, ground truth (if available), and coherence
    
    Args:
        statement_to_original_index: Optional mapping from resampled statement index to original index.
                                    Used when statements have been resampled and ground truth indices need to be mapped.
    """
    import csv
    
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    # Create CSV with header: Statement, Question, Ground_Truth (if available), Model1, Model2, Model3, Coherence
    fieldnames = ['Statement', 'Question']
    if has_ground_truth:
        fieldnames.append('Ground_Truth')
    fieldnames.extend([model_name for model_name in model_names])
    fieldnames.append('Coherence')
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write results for each statement-question pair
        for stmt_idx, statement in enumerate(statements):
            for question_id in question_ids:
                row = {
                    'Statement': statement,
                    'Question': question_id
                }
                
                # Get ground truth (if available)
                if has_ground_truth:
                    # Use mapping if provided (for resampled statements), otherwise use stmt_idx directly
                    original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
                    ground_truth = GROUND_TRUTH.get(original_idx, {}).get(question_id, 'unknown')
                    row['Ground_Truth'] = ground_truth
                
                # Get classification from each model
                responses = []
                for model_name in model_names:
                    model_result = all_results[model_name].get(stmt_idx, {})
                    response = model_result.get(question_id, 'error')
                    row[model_name] = response
                    responses.append(response)
                
                # Calculate coherence
                coherence = calculate_coherence(responses)
                row['Coherence'] = coherence
                
                writer.writerow(row)
    
    print(f"Results saved to {filename}")

def save_results_json(all_results: Dict[str, Dict], statements: List[str], questions: Dict, 
                      filename: str = "results.json"):
    """
    Save results to JSON file with comments as keys and all model labels grouped together.
    Structure: {comment_text: {model_name: {question_id: response}}}
    
    Args:
        all_results: Dictionary with structure {model_name: {statement_index: {question_id: response}}}
        statements: List of statement/comment strings
        questions: Dictionary of questions
        filename: Output JSON filename
    """
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    # Build JSON structure: {comment_text: {model_name: {question_id: response}}}
    results_json = {}
    
    for stmt_idx, statement in enumerate(statements):
        comment_results = {}
        for model_name in model_names:
            model_result = all_results[model_name].get(stmt_idx, {})
            question_responses = {}
            for question_id in question_ids:
                response = model_result.get(question_id, 'error')
                question_responses[question_id] = response
            comment_results[model_name] = question_responses
        results_json[statement] = comment_results
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {filename}")

def save_results_json_relevant_only(all_results: Dict[str, Dict], statements: List[str], questions: Dict,
                                    relevance_results: Dict[int, Dict[str, str]], 
                                    filename: str = "results_relevant_only.json"):
    """
    Save results to JSON file, but only for statements that are relevant to at least one question.
    Structure: {comment_text: {model_name: {question_id: response}}}
    
    Args:
        all_results: Dictionary with structure {model_name: {statement_index: {question_id: response}}}
        statements: List of statement/comment strings
        questions: Dictionary of questions
        relevance_results: Dictionary with structure {statement_index: {question_id: 'yes'/'no'/'error'}}
        filename: Output JSON filename
    """
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    # Find relevant statement indices (statements with at least one "yes" in relevance_results)
    relevant_stmt_indices = set()
    for stmt_idx in relevance_results:
        for question_id in question_ids:
            if relevance_results[stmt_idx].get(question_id) == 'yes':
                relevant_stmt_indices.add(stmt_idx)
                break  # Only need one "yes" to be considered relevant
    
    if not relevant_stmt_indices:
        print(f"No relevant statements found. Skipping {filename}")
        return
    
    # Build JSON structure: {comment_text: {model_name: {question_id: response}}}
    results_json = {}
    
    for stmt_idx in relevant_stmt_indices:
        if stmt_idx >= len(statements):
            continue
        statement = statements[stmt_idx]
        comment_results = {}
        for model_name in model_names:
            model_result = all_results[model_name].get(stmt_idx, {})
            question_responses = {}
            for question_id in question_ids:
                response = model_result.get(question_id, 'error')
                question_responses[question_id] = response
            comment_results[model_name] = question_responses
        results_json[statement] = comment_results
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    print(f"Relevant-only results saved to {filename} ({len(relevant_stmt_indices)} relevant statements)")

def save_results_csv_relevant_only(all_results: Dict[str, Dict], statements: List[str], questions: Dict, 
                                   relevance_results: Dict[int, Dict[str, str]], 
                                   filename: str = "sample_local_llm_results_relevant_only.csv", 
                                   has_ground_truth: bool = True, 
                                   statement_to_original_index: Dict[int, int] = None):
    """
    Save results to CSV file, but only for statements that are relevant to at least one question.
    A statement is considered relevant if at least one question has "yes" in relevance_results.
    
    Args:
        relevance_results: Dictionary with structure {statement_index: {question_id: 'yes'/'no'/'error'}}
        statement_to_original_index: Optional mapping from resampled statement index to original index.
                                    Used when statements have been resampled and ground truth indices need to be mapped.
    """
    import csv
    
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    # Find relevant statement indices (statements with at least one "yes" in relevance_results)
    relevant_stmt_indices = set()
    for stmt_idx in relevance_results:
        for question_id in question_ids:
            if relevance_results[stmt_idx].get(question_id) == 'yes':
                relevant_stmt_indices.add(stmt_idx)
                break  # Only need one "yes" to be considered relevant
    
    if not relevant_stmt_indices:
        print(f"No relevant statements found. Skipping {filename}")
        return
    
    # Create CSV with header: Statement, Question, Ground_Truth (if available), Model1, Model2, Model3, Coherence
    fieldnames = ['Statement', 'Question']
    if has_ground_truth:
        fieldnames.append('Ground_Truth')
    fieldnames.extend([model_name for model_name in model_names])
    fieldnames.append('Coherence')
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write results only for relevant statements
        for stmt_idx in relevant_stmt_indices:
            if stmt_idx >= len(statements):
                continue
            statement = statements[stmt_idx]
            for question_id in question_ids:
                row = {
                    'Statement': statement,
                    'Question': question_id
                }
                
                # Get ground truth (if available)
                if has_ground_truth:
                    # Use mapping if provided (for resampled statements), otherwise use stmt_idx directly
                    original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
                    ground_truth = GROUND_TRUTH.get(original_idx, {}).get(question_id, 'unknown')
                    row['Ground_Truth'] = ground_truth
                
                # Get classification from each model
                responses = []
                for model_name in model_names:
                    model_result = all_results[model_name].get(stmt_idx, {})
                    response = model_result.get(question_id, 'error')
                    row[model_name] = response
                    responses.append(response)
                
                # Calculate coherence
                coherence = calculate_coherence(responses)
                row['Coherence'] = coherence
                
                writer.writerow(row)
    
    print(f"Relevant-only results saved to {filename} ({len(relevant_stmt_indices)} relevant statements)")

def calculate_metrics(all_results: Dict[str, Dict], statements: List[str], questions: Dict, has_ground_truth: bool = True, statement_to_original_index: Dict[int, int] = None) -> Dict:
    """
    Calculate accuracy (if ground truth available) and coherence metrics
    
    Args:
        statement_to_original_index: Optional mapping from resampled statement index to original index.
                                    Used when statements have been resampled and ground truth indices need to be mapped.
    """
    question_ids = list(questions.keys())
    model_names = list(all_results.keys())
    
    total_pairs = len(statements) * len(question_ids)
    
    # Initialize counters
    model_correct = {model_name: 0 for model_name in model_names}
    majority_correct = 0
    coherence_correct = []
    coherence_wrong = []
    coherence_all = []
    
    # Track accuracy by question type
    question_stats = {q_id: {'total': 0, 'correct': {model_name: 0 for model_name in model_names}, 'majority_correct': 0} 
                      for q_id in question_ids}
    
    # Process each statement-question pair
    for stmt_idx in range(len(statements)):
        for question_id in question_ids:
            # Get responses from all models
            responses = []
            for model_name in model_names:
                model_result = all_results[model_name].get(stmt_idx, {})
                response = model_result.get(question_id, 'error')
                responses.append(response)
            
            # Calculate coherence (always calculated)
            coherence = calculate_coherence(responses)
            coherence_all.append(coherence)
            
            # Accuracy calculations only if ground truth is available
            if has_ground_truth:
                # Use mapping if provided (for resampled statements), otherwise use stmt_idx directly
                original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
                ground_truth = GROUND_TRUTH.get(original_idx, {}).get(question_id, 'unknown')
                if ground_truth == 'unknown':
                    continue
                
                # Update question stats
                question_stats[question_id]['total'] += 1
                
                # Check if each model is correct
                for model_name, response in zip(model_names, responses):
                    if response == ground_truth:
                        model_correct[model_name] += 1
                        question_stats[question_id]['correct'][model_name] += 1
                
                # Check majority vote
                majority = get_majority_vote(responses)
                if majority == ground_truth:
                    majority_correct += 1
                    question_stats[question_id]['majority_correct'] += 1
                    coherence_correct.append(coherence)
                else:
                    coherence_wrong.append(coherence)
    
    # Calculate percentages
    metrics = {
        'total_pairs': total_pairs,
        'has_ground_truth': has_ground_truth,
        'total_coherence': sum(coherence_all) / len(coherence_all) if coherence_all else 0,
    }
    
    # Add accuracy metrics only if ground truth is available
    if has_ground_truth:
        metrics['model_accuracy'] = {model_name: (model_correct[model_name] / total_pairs * 100) 
                                       for model_name in model_names}
        metrics['majority_accuracy'] = (majority_correct / total_pairs * 100) if total_pairs > 0 else 0
        metrics['coherence_when_correct'] = sum(coherence_correct) / len(coherence_correct) if coherence_correct else 0
        metrics['coherence_when_wrong'] = sum(coherence_wrong) / len(coherence_wrong) if coherence_wrong else 0
        
        # Calculate accuracy by question
        metrics['accuracy_by_question'] = {}
        for question_id, stats in question_stats.items():
            if stats['total'] > 0:
                metrics['accuracy_by_question'][question_id] = {
                    'total': stats['total'],
                    'model_accuracy': {model_name: (stats['correct'][model_name] / stats['total'] * 100) 
                                       for model_name in model_names},
                    'majority_accuracy': (stats['majority_correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                }
    
    return metrics

def save_metrics_csv(metrics: Dict, filename: str = "sample_local_llm_metrics.csv"):
    """Save metrics to CSV file"""
    import csv
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Pairs', metrics['total_pairs']])
        writer.writerow(['Total Coherence (%)', round(metrics['total_coherence'], 2)])
        
        # Only include accuracy metrics if ground truth is available
        if metrics.get('has_ground_truth', True):
            writer.writerow(['Majority Accuracy (%)', round(metrics['majority_accuracy'], 2)])
            writer.writerow(['Coherence When Correct (%)', round(metrics['coherence_when_correct'], 2)])
            writer.writerow(['Coherence When Wrong (%)', round(metrics['coherence_when_wrong'], 2)])
            
            for model_name, accuracy in metrics['model_accuracy'].items():
                writer.writerow([f'{model_name} Accuracy (%)', round(accuracy, 2)])
    
    print(f"Metrics saved to {filename}")

def print_metrics(metrics: Dict):
    """Print condensed metrics"""
    print("\n" + "="*70)
    print("CLASSIFICATION METRICS (Step 2)")
    print("="*70)
    
    if metrics.get('has_ground_truth', True):
        model_names = list(metrics['model_accuracy'].keys())
        
        # Overall accuracy (condensed)
        print(f"{'Overall Accuracy':<35} " + " ".join([f"{m.split('/')[-1][:12]:>12}" for m in model_names]) + f" {'Majority':>10}")
        print("-" * 70)
        row = f"{'':<35} "
        for model_name in model_names:
            row += f"{metrics['model_accuracy'][model_name]:>11.1f}%"
        row += f" {metrics['majority_accuracy']:>9.1f}%"
        print(row)
        
        # Accuracy by question (condensed table)
        if 'accuracy_by_question' in metrics and metrics['accuracy_by_question']:
            print("\n" + "-" * 70)
            print("ACCURACY BY QUESTION")
            print("-" * 70)
            print(f"{'Question':<35} " + " ".join([f"{m.split('/')[-1][:12]:>12}" for m in model_names]) + f" {'Majority':>10}")
            print("-" * 70)
            
            for question_id, q_metrics in sorted(metrics['accuracy_by_question'].items()):
                row = f"{question_id:<35} "
                for model_name in model_names:
                    acc = q_metrics['model_accuracy'].get(model_name, 0)
                    row += f"{acc:>11.1f}%"
                row += f" {q_metrics['majority_accuracy']:>9.1f}%"
                print(row)
        
        # Coherence (condensed)
        print("\n" + "-" * 70)
        print(f"Coherence: {metrics['total_coherence']:.1f}% | "
              f"Correct: {metrics['coherence_when_correct']:.1f}% | "
              f"Wrong: {metrics['coherence_when_wrong']:.1f}%")
    else:
        print(f"Coherence: {metrics['total_coherence']:.1f}%")
        print("(No ground truth - accuracy not calculated)")
    
    print("="*70 + "\n")

def load_survey_questions_by_sector() -> Dict[str, Dict]:
    """Load survey questions from JSON file organized by sector"""
    if not os.path.exists('survey_question.json'):
        print("Error: survey_question.json not found.")
        return {}
    
    with open('survey_question.json', 'r', encoding='utf-8') as f:
        survey_data = json.load(f)
    
    # Normalize sector names (handle case differences)
    sector_mapping = {
        'food': 'Food',
        'housing': 'Housing',
        'transport': 'Transport',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    
    normalized_data = {}
    for sector, questions in survey_data.items():
        # Normalize to title case
        normalized_sector = sector.capitalize()
        # Ensure each question has the sector field set
        for question_id, question_info in questions.items():
            if 'sector' not in question_info:
                question_info['sector'] = normalized_sector
        normalized_data[normalized_sector] = questions
    
    return normalized_data

def get_questions_for_sector(sector: str) -> Dict:
    """
    Get survey questions for a specific sector only (no cross-sector questions)
    
    Args:
        sector: Sector name (Food, Transport, Housing, or lowercase variants)
    
    Returns:
        Dictionary of questions for the specified sector only
    """
    survey_by_sector = load_survey_questions_by_sector()
    
    # Normalize sector name
    sector_mapping = {
        'food': 'Food',
        'transport': 'Transport',
        'housing': 'Housing',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    normalized_sector = sector_mapping.get(sector.lower(), sector.capitalize())
    
    if normalized_sector not in survey_by_sector:
        print(f"Warning: Sector '{sector}' not found. Available sectors: {', '.join(survey_by_sector.keys())}")
        return {}
    
    # Get questions for this sector and filter to ensure they all have the correct sector
    sector_questions = survey_by_sector[normalized_sector].copy()
    
    # Double-check: filter out any questions that don't match the sector
    filtered_questions = {}
    for question_id, question_info in sector_questions.items():
        # Ensure sector field matches
        if question_info.get('sector', '').capitalize() == normalized_sector:
            filtered_questions[question_id] = question_info
        else:
            print(f"Warning: Question {question_id} has sector '{question_info.get('sector')}' but expected '{normalized_sector}'. Skipping.")
    
    return filtered_questions

def calculate_relevance_accuracy(all_relevance_results: Dict[str, Dict], statements: List[str], questions: Dict, 
                                 has_ground_truth: bool = True, statement_to_original_index: Dict[int, int] = None) -> Dict:
    """
    Calculate accuracy of relevance detection (step 1), especially for irrelevant statements.
    A statement is irrelevant if all questions have ground truth "no".
    """
    if not has_ground_truth:
        return {}
    
    question_ids = list(questions.keys())
    model_names = list(all_relevance_results.keys())
    
    # Identify irrelevant statements (all questions marked as "no" in ground truth)
    irrelevant_statements = []
    for stmt_idx in range(len(statements)):
        original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
        ground_truth_for_stmt = GROUND_TRUTH.get(original_idx, {})
        # Check if all questions are "no" (irrelevant)
        all_no = all(ground_truth_for_stmt.get(q_id, 'unknown') == 'no' for q_id in question_ids)
        if all_no:
            irrelevant_statements.append(stmt_idx)
    
    # Calculate accuracy for detecting irrelevant statements
    irrelevant_detection = {model_name: {'correct': 0, 'total': len(irrelevant_statements)} 
                           for model_name in model_names}
    
    for stmt_idx in irrelevant_statements:
        for model_name in model_names:
            # Check if model correctly identified as irrelevant (all "no")
            relevance_results = all_relevance_results[model_name].get(stmt_idx, {})
            all_detected_no = all(relevance_results.get(q_id) == 'no' for q_id in question_ids)
            if all_detected_no:
                irrelevant_detection[model_name]['correct'] += 1
    
    # Calculate overall relevance accuracy (per question)
    question_relevance_accuracy = {q_id: {'total': 0, 'correct': {model_name: 0 for model_name in model_names}}
                                   for q_id in question_ids}
    
    for stmt_idx in range(len(statements)):
        original_idx = statement_to_original_index.get(stmt_idx, stmt_idx) if statement_to_original_index else stmt_idx
        ground_truth_for_stmt = GROUND_TRUTH.get(original_idx, {})
        
        for question_id in question_ids:
            gt = ground_truth_for_stmt.get(question_id, 'unknown')
            if gt == 'unknown':
                continue
            
            question_relevance_accuracy[question_id]['total'] += 1
            
            for model_name in model_names:
                predicted = all_relevance_results[model_name].get(stmt_idx, {}).get(question_id)
                if predicted == gt:
                    question_relevance_accuracy[question_id]['correct'][model_name] += 1
    
    return {
        'irrelevant_detection': irrelevant_detection,
        'question_relevance_accuracy': question_relevance_accuracy,
        'irrelevant_count': len(irrelevant_statements)
    }

def print_relevance_percentages(all_relevance_results: Dict[str, Dict], questions: Dict, statements: List[str] = None,
                                has_ground_truth: bool = True, statement_to_original_index: Dict[int, int] = None):
    """Print condensed relevance percentages and accuracy"""
    if statements is None:
        statements = SAMPLE_STATEMENTS
    
    question_ids = list(questions.keys())
    model_names = list(all_relevance_results.keys())
    
    print("\n" + "="*70)
    print("RELEVANCE CHECK (Step 1)")
    print("="*70)
    
    # Print relevance percentages (condensed)
    print(f"{'Question':<35} " + " ".join([f"{m.split('/')[-1][:12]:>12}" for m in model_names]))
    print("-" * 70)
    
    for question_id in question_ids:
        row = f"{question_id:<35} "
        for model_name in model_names:
            relevant_count = sum(1 for stmt_idx in all_relevance_results[model_name] 
                               if all_relevance_results[model_name][stmt_idx].get(question_id) == 'yes')
            percentage = (relevant_count / len(statements) * 100) if len(statements) > 0 else 0
            row += f"{percentage:>11.1f}%"
        print(row)
    
    # Print relevance accuracy if ground truth available
    if has_ground_truth:
        rel_accuracy = calculate_relevance_accuracy(all_relevance_results, statements, questions, 
                                                    has_ground_truth, statement_to_original_index)
        
        if rel_accuracy and rel_accuracy.get('irrelevant_count', 0) > 0:
            print("\n" + "-" * 70)
            print("IRRELEVANT STATEMENT DETECTION ACCURACY")
            print("-" * 70)
            print(f"Irrelevant statements: {rel_accuracy['irrelevant_count']}")
            for model_name in model_names:
                stats = rel_accuracy['irrelevant_detection'][model_name]
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"  {model_name.split('/')[-1][:30]:<30} {acc:>5.1f}% ({stats['correct']}/{stats['total']})")
        
        # Print overall relevance accuracy per question (condensed)
        if rel_accuracy.get('question_relevance_accuracy'):
            print("\n" + "-" * 70)
            print("RELEVANCE ACCURACY BY QUESTION")
            print("-" * 70)
            print(f"{'Question':<35} " + " ".join([f"{m.split('/')[-1][:12]:>12}" for m in model_names]))
            print("-" * 70)
            for q_id in question_ids:
                row = f"{q_id:<35} "
                total = rel_accuracy['question_relevance_accuracy'][q_id]['total']
                for model_name in model_names:
                    correct = rel_accuracy['question_relevance_accuracy'][q_id]['correct'][model_name]
                    acc = (correct / total * 100) if total > 0 else 0
                    row += f"{acc:>11.1f}%"
                print(row)
    
    print("="*70 + "\n")

# ============================================================================
# End of standalone code from sample_local_llm.py
# ============================================================================

# vLLM Server Configuration - Multiple Models Available
# Load model configuration from JSON file
with open('schema/local_LLM_api_from_vLLM.json', 'r') as f:
    vllm_config = json.load(f)
    AVAILABLE_MODELS = vllm_config['available_models']
    DEFAULT_MODEL_KEY = vllm_config.get('default_model_key', '1')
VLLM_BASE_URL = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["base_url"]
VLLM_MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["model_name"]

# Performance tuning for continuous batching
# Optimized for RTX 5090 32GB GPU with Qwen2.5-3B-Instruct model
# Higher concurrency = more requests batched together by vLLM
# vLLM automatically batches concurrent requests in-flight for maximum GPU utilization
# 
# RTX 5090 32GB settings:
#   - 32GB VRAM allows for very high concurrency
#   - 3B model is relatively small, can handle many concurrent requests
#   - Can maximize throughput with high concurrent request limits
MAX_CONCURRENT_REQUESTS = 450  # Optimized for 32GB GPU - can handle high concurrency
REQUEST_TIMEOUT = 60  # Increased timeout for high concurrency scenarios

# Load survey questions (default - will be reloaded in main if flag is set)
SURVEY_QUESTIONS = load_survey_questions(use_real_survey_questions=False)

def model_supports_system_messages(model_name: str) -> bool:
    """
    Check if a model supports system messages in its chat template.
    Some models (e.g., Gemma, some Mistral variants) only accept user/assistant roles.
    
    Args:
        model_name: The model name/ID from the server
        
    Returns:
        True if model supports system messages, False otherwise
    """
    model_lower = model_name.lower()
    # Gemma models don't support system messages
    if 'gemma' in model_lower:
        return False
    # Some Mistral variants may not support system messages
    # Add specific checks here if needed
    # For now, assume Mistral supports it unless we know otherwise
    return True

def build_messages_array(prompt: str, model_name: str, system_content: str = None) -> List[Dict]:
    """
    Build messages array for API call, handling models that don't support system messages.
    
    Args:
        prompt: The user prompt
        model_name: The model name/ID
        system_content: Optional system message content
        
    Returns:
        List of message dictionaries
    """
    if system_content is None:
        system_content = "Please act as an expert annotator for survey question classification. Please always respond with valid JSON."
    
    if model_supports_system_messages(model_name):
        # Model supports system messages - use separate system role
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
    else:
        # Model doesn't support system messages - merge into user message
        combined_prompt = f"{system_content}\n\n{prompt}"
        return [
            {"role": "user", "content": combined_prompt}
        ]

async def call_vllm_api_async(session: aiohttp.ClientSession, prompt: str, 
                              base_url: str = VLLM_BASE_URL, 
                              model_name: str = VLLM_MODEL_NAME) -> Tuple[str, Dict]:
    """
    Async call to vLLM API using OpenAI-compatible endpoint
    Optimized for continuous batching - vLLM will batch concurrent requests automatically
    
    Args:
        session: aiohttp ClientSession for async requests
        prompt: The prompt to send
        base_url: Base URL for the API (default: localhost:8000)
        model_name: Model name as served by vLLM (default: qwen2.5-3b)
    
    Returns:
        Tuple of (response_text, token_usage_dict) where token_usage_dict contains:
        {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            # Try to extract JSON from response (in case there's extra text)
            content_clean = content.strip()
            # Remove markdown code blocks if present
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            # Validate against schema
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    # Build messages array (handles models that don't support system messages)
    messages = build_messages_array(prompt, model_name)
    
    # Try with JSON schema enforcement first
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    # Fallback payload without schema (if API doesn't support it)
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    # Helper to extract token usage from response
    def extract_token_usage(result: Dict) -> Dict:
        """Extract token usage from vLLM API response"""
        if 'usage' in result:
            usage = result['usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Try with schema first
    try:
        async with session.post(url, json=payload_with_schema, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
            content = result['choices'][0]['message']['content'].strip()
            parsed = parse_and_validate_response(content)
            token_usage = extract_token_usage(result)
            if parsed != "error":
                return parsed, token_usage
    except aiohttp.ClientResponseError as e:
        # If 400 error, might be unsupported response_format, try without
        if e.status == 400:
            try:
                async with session.post(url, json=payload_without_schema, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    parsed = parse_and_validate_response(content)
                    token_usage = extract_token_usage(result)
                    return parsed, token_usage
            except Exception:
                return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        else:
            return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    except Exception:
        return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

async def call_vllm_api_async_with_debug(session: aiohttp.ClientSession, prompt: str, 
                                         base_url: str = VLLM_BASE_URL, 
                                         model_name: str = VLLM_MODEL_NAME,
                                         debug: bool = False) -> Tuple[str, Dict, str]:
    """
    Async call to vLLM API with debug support - returns raw content for inspection
    
    Returns:
        Tuple of (parsed_response, token_usage_dict, raw_content)
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    messages = build_messages_array(prompt, model_name)
    
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    def extract_token_usage(result: Dict) -> Dict:
        """Extract token usage from vLLM API response"""
        if 'usage' in result:
            usage = result['usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Try with schema first
    try:
        async with session.post(url, json=payload_with_schema, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            result = await response.json()
            content = result['choices'][0]['message']['content'].strip()
            raw_content = content  # Store raw content
            parsed = parse_and_validate_response(content)
            token_usage = extract_token_usage(result)
            if parsed != "error":
                return parsed, token_usage, raw_content
    except aiohttp.ClientResponseError as e:
        if e.status == 400:
            try:
                async with session.post(url, json=payload_without_schema, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()
                    result = await response.json()
                    content = result['choices'][0]['message']['content'].strip()
                    raw_content = content  # Store raw content
                    parsed = parse_and_validate_response(content)
                    token_usage = extract_token_usage(result)
                    return parsed, token_usage, raw_content
            except Exception:
                return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
        else:
            return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
    except Exception:
        return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"
    
    return "error", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}, "error"

def call_vllm_api(prompt: str, base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME) -> str:
    """
    Call vLLM API using OpenAI-compatible endpoint
    
    Args:
        prompt: The prompt to send
        base_url: Base URL for the API (default: localhost:8000)
        model_name: Model name as served by vLLM (default: qwen2.5-3b)
    
    Returns:
        Response text from the model
    """
    endpoint = api_config['config']['endpoints']['chat_completions']['path']
    url = f"{base_url}{endpoint}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Build messages array (handles models that don't support system messages)
    messages = build_messages_array(prompt, model_name)
    
    # Try with JSON schema enforcement first
    payload_with_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens'],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_response",
                "strict": True,
                "schema": RESPONSE_SCHEMA
            }
        }
    }
    
    # Fallback payload without schema (if API doesn't support it)
    payload_without_schema = {
        "model": model_name,
        "messages": messages,
        "temperature": api_config['config']['default_temperature'],
        "max_tokens": api_config['config']['default_max_tokens']
    }
    
    def parse_and_validate_response(content: str) -> str:
        """Parse and validate JSON response against schema"""
        try:
            # Try to extract JSON from response (in case there's extra text)
            content_clean = content.strip()
            # Remove markdown code blocks if present
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            json_response = json.loads(content_clean)
            # Validate against schema
            validate(instance=json_response, schema=RESPONSE_SCHEMA)
            return json_response['relevant'].lower()
        except json.JSONDecodeError:
            return "error"
        except ValidationError:
            return "error"
    
    # Try with schema first
    try:
        response = requests.post(url, json=payload_with_schema, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        parsed = parse_and_validate_response(content)
        if parsed != "error":
            return parsed
    except requests.exceptions.HTTPError as e:
        # If 400 error, might be unsupported response_format, try without
        if e.response.status_code == 400:
            try:
                # Try without schema
                response = requests.post(url, json=payload_without_schema, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return parse_and_validate_response(content)
            except Exception as e2:
                print(f"Error calling vLLM API (fallback): {e2}")
                return "error"
        else:
            print(f"HTTP error calling vLLM API: {e}")
            return "error"
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return "error"
    
    return "error"

async def check_relevance_vllm_async(session: aiohttp.ClientSession, statement: str, question_id: str, 
                                    question_info: Dict, base_url: str = VLLM_BASE_URL, 
                                    model_name: str = VLLM_MODEL_NAME) -> Tuple[str, str, Dict]:
    """
    First pass: Check if statement is relevant to question (vLLM async version)
    
    Returns:
        Tuple of (question_id, response, token_usage) where response is 'yes', 'no', or 'error'
    """
    prompt = create_relevance_prompt(statement, question_info)
    response, token_usage = await call_vllm_api_async(session, prompt, base_url, model_name)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes', token_usage)
    elif 'no' in response or response == 'n':
        return (question_id, 'no', token_usage)
    else:
        return (question_id, 'error', token_usage)

async def check_relevance_grouped_vllm_async(session: aiohttp.ClientSession, statement: str, 
                                            grouped_questions: List[Tuple[str, Dict]], 
                                            base_url: str = VLLM_BASE_URL, 
                                            model_name: str = VLLM_MODEL_NAME) -> Dict[str, Tuple[str, Dict]]:
    """
    First pass: Check if statement is relevant to a group of related questions (vLLM async version).
    Returns the same relevance result for all questions in the group.
    
    Returns:
        Dictionary mapping question_id -> (response, token_usage) where response is 'yes'/'no'/'error'
    """
    prompt = create_relevance_prompt_grouped(statement, grouped_questions)
    response, token_usage = await call_vllm_api_async(session, prompt, base_url, model_name)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        result = 'yes'
    elif 'no' in response or response == 'n':
        result = 'no'
    else:
        result = 'error'
    
    # Return same result for all questions in the group
    return {question_id: (result, token_usage) for question_id, _ in grouped_questions}

async def classify_statement_against_question_vllm_async(session: aiohttp.ClientSession, statement: str, 
                                                         question_id: str, question_info: Dict, 
                                                         base_url: str = VLLM_BASE_URL, 
                                                         model_name: str = VLLM_MODEL_NAME) -> Tuple[str, str, Dict]:
    """
    Second pass: Classify a relevant statement against a survey question (vLLM async version)
    
    Returns:
        Tuple of (question_id, response, token_usage) where response is 'yes', 'no', or 'error'
    """
    prompt = create_classification_prompt(statement, question_info)
    response, token_usage = await call_vllm_api_async(session, prompt, base_url, model_name)
    
    # Normalize response to yes/no
    if 'yes' in response or response == 'y':
        return (question_id, 'yes', token_usage)
    elif 'no' in response or response == 'n':
        return (question_id, 'no', token_usage)
    else:
        return (question_id, 'error', token_usage)

async def check_relevance_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                         stmt_idx: int, statement: str, question_id: str, 
                                         question_info: Dict, base_url: str, model_name: str,
                                         pbar: tqdm, token_counter: Dict) -> Tuple[int, str, str]:
    """First pass: Check relevance of a single statement-question pair (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            _, response, token_usage = await check_relevance_vllm_async(session, statement, question_id, question_info, base_url, model_name)
            # Accumulate token counts
            token_counter['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
            token_counter['completion_tokens'] += token_usage.get('completion_tokens', 0)
            token_counter['total_tokens'] += token_usage.get('total_tokens', 0)
            pbar.update(1)
            return (stmt_idx, question_id, response)
        except Exception:
            pbar.update(1)
            return (stmt_idx, question_id, 'error')

async def check_relevance_grouped_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                                  stmt_idx: int, statement: str, grouped_questions: List[Tuple[str, Dict]], 
                                                  base_url: str, model_name: str,
                                                  pbar: tqdm, token_counter: Dict) -> List[Tuple[int, str, str]]:
    """First pass: Check relevance of a statement to a group of related questions (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            grouped_results = await check_relevance_grouped_vllm_async(session, statement, grouped_questions, base_url, model_name)
            # Accumulate token counts (use first result's token usage, they're all the same)
            if grouped_results:
                first_token_usage = next(iter(grouped_results.values()))[1]
                token_counter['prompt_tokens'] += first_token_usage.get('prompt_tokens', 0)
                token_counter['completion_tokens'] += first_token_usage.get('completion_tokens', 0)
                token_counter['total_tokens'] += first_token_usage.get('total_tokens', 0)
            
            # Return list of (stmt_idx, question_id, response) tuples
            results = [(stmt_idx, question_id, response) for question_id, (response, _) in grouped_results.items()]
            pbar.update(len(results))  # Update progress bar for all questions in group
            return results
        except Exception:
            # Mark all questions in group as error
            results = [(stmt_idx, question_id, 'error') for question_id, _ in grouped_questions]
            pbar.update(len(results))
            return results

async def classify_single_task_vllm_async(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                                         stmt_idx: int, statement: str, question_id: str, 
                                         question_info: Dict, base_url: str, model_name: str,
                                         pbar: tqdm, token_counter: Dict) -> Tuple[int, str, str]:
    """Second pass: Classify a relevant statement-question pair (vLLM async version)"""
    async with semaphore:  # Limit concurrent requests
        try:
            _, response, token_usage = await classify_statement_against_question_vllm_async(
                session, statement, question_id, question_info, base_url, model_name
            )
            # Accumulate token counts
            token_counter['prompt_tokens'] += token_usage.get('prompt_tokens', 0)
            token_counter['completion_tokens'] += token_usage.get('completion_tokens', 0)
            token_counter['total_tokens'] += token_usage.get('total_tokens', 0)
            pbar.update(1)
            return (stmt_idx, question_id, response)
        except Exception:
            pbar.update(1)
            return (stmt_idx, question_id, 'error')

async def first_pass_relevance_vllm_async(statements: List[str], questions: Dict, 
                                         base_url: str = VLLM_BASE_URL, 
                                         model_name: str = VLLM_MODEL_NAME,
                                         max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> Tuple[Dict, Dict]:
    """
    First pass: Determine relevance of all statements to all questions (vLLM async version)
    Uses grouped questions to reduce API calls (e.g., Help/Hurt variants checked together).
    Uses async requests with semaphore for controlled concurrency.
    vLLM's continuous batching will automatically batch concurrent requests together.
    
    Returns:
        Tuple of (results_dict, token_usage_dict) where:
        - results_dict: {statement_index: {question_id: 'yes'/'no'/'error'}}
        - token_usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    results = {}
    
    # Group related questions by base topic
    question_groups = group_related_questions(questions)
    
    # Initialize results structure
    for stmt_idx in range(len(statements)):
        results[stmt_idx] = {}
    
    # Calculate total tasks for progress bar (still one per statement-question pair for display)
    total_tasks = len(statements) * len(questions)
    
    # Token counter (shared across tasks)
    token_counter = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Create semaphore to limit concurrent requests (allows vLLM to batch efficiently)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session with connection pooling for better performance
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    with tqdm(total=total_tasks, desc=f"{model_name[:30]:<30} [Pass 1: Relevance]", unit="task") as pbar:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks (one per statement-group pair)
            tasks = []
            for stmt_idx, statement in enumerate(statements):
                for grouped_questions in question_groups.values():
                    task = check_relevance_grouped_task_vllm_async(
                        session, semaphore, stmt_idx, statement, grouped_questions, 
                        base_url, model_name, pbar, token_counter
                    )
                    tasks.append(task)
            
            # Execute all tasks concurrently (vLLM will batch them)
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in task_results:
                if isinstance(result, Exception):
                    continue
                # Result is a list of (stmt_idx, question_id, response) tuples
                for stmt_idx, question_id, response in result:
                    results[stmt_idx][question_id] = response
    
    return results, token_counter

async def second_pass_classification_vllm_async(statements: List[str], questions: Dict, 
                                                relevance_results: Dict,
                                                base_url: str = VLLM_BASE_URL, 
                                                model_name: str = VLLM_MODEL_NAME,
                                                max_concurrent: int = MAX_CONCURRENT_REQUESTS) -> Tuple[Dict, Dict]:
    """
    Second pass: Detailed classification only for statements marked as relevant (vLLM async version)
    Uses async requests with semaphore for controlled concurrency.
    vLLM's continuous batching will automatically batch concurrent requests together.
    
    Args:
        relevance_results: Results from first pass showing which statements are relevant to which questions
    
    Returns:
        Tuple of (results_dict, token_usage_dict) where:
        - results_dict: {statement_index: {question_id: 'yes'/'no'/'error'}}
        - token_usage_dict: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
    """
    results = {}
    
    # Prepare tasks only for relevant pairs
    tasks = []
    for stmt_idx, statement in enumerate(statements):
        results[stmt_idx] = {}
        for question_id, question_info in questions.items():
            # Only process if marked as relevant in first pass
            if relevance_results.get(stmt_idx, {}).get(question_id) == 'yes':
                tasks.append((stmt_idx, statement, question_id, question_info))
            else:
                # Mark as not relevant (from first pass)
                results[stmt_idx][question_id] = 'no'
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        return results, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Token counter (shared across tasks)
    token_counter = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create aiohttp session with connection pooling
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    
    with tqdm(total=total_tasks, desc=f"{model_name[:30]:<30} [Pass 2: Classification]", unit="task") as pbar:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create all tasks
            async_tasks = []
            for stmt_idx, statement, question_id, question_info in tasks:
                task = classify_single_task_vllm_async(
                    session, semaphore, stmt_idx, statement, question_id, 
                    question_info, base_url, model_name, pbar, token_counter
                )
                async_tasks.append(task)
            
            # Execute all tasks concurrently (vLLM will batch them)
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Process results
            for result in task_results:
                if isinstance(result, Exception):
                    continue
                stmt_idx, question_id, response = result
                results[stmt_idx][question_id] = response
    
    return results, token_counter

# Synchronous wrapper functions for backward compatibility
def first_pass_relevance_vllm(statements: List[str], questions: Dict, 
                              base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME,
                              max_workers: int = None) -> Tuple[Dict, Dict]:
    """
    Synchronous wrapper for async first pass (for backward compatibility)
    Note: For maximum throughput, use first_pass_relevance_vllm_async directly
    
    Returns:
        Tuple of (results_dict, token_usage_dict)
    """
    if max_workers is None:
        max_workers = MAX_CONCURRENT_REQUESTS
    return asyncio.run(first_pass_relevance_vllm_async(statements, questions, base_url, model_name, max_workers))

def second_pass_classification_vllm(statements: List[str], questions: Dict, relevance_results: Dict,
                                   base_url: str = VLLM_BASE_URL, model_name: str = VLLM_MODEL_NAME,
                                   max_workers: int = None) -> Tuple[Dict, Dict]:
    """
    Synchronous wrapper for async second pass (for backward compatibility)
    Note: For maximum throughput, use second_pass_classification_vllm_async directly
    
    Returns:
        Tuple of (results_dict, token_usage_dict)
    """
    if max_workers is None:
        max_workers = MAX_CONCURRENT_REQUESTS
    return asyncio.run(second_pass_classification_vllm_async(statements, questions, relevance_results, base_url, model_name, max_workers))

def get_model_id_from_server(base_url: str) -> str:
    """Get the actual model ID from the vLLM server"""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        if 'data' in models and len(models['data']) > 0:
            return models['data'][0]['id']
        return None
    except Exception as e:
        return None

def test_vllm_connection(base_url: str = None) -> bool:
    """Test if vLLM server is accessible and show actual model ID"""
    if base_url is None:
        base_url = VLLM_BASE_URL
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        actual_model_id = get_model_id_from_server(base_url)
        print(f"✓ vLLM server is accessible at {base_url}")
        if actual_model_id:
            print(f"  Actual model ID: {actual_model_id}")
        print(f"  Available models: {models}")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to vLLM server at {base_url}")
        print(f"  Error: {e}")
        print(f"  Make sure the server is running on the specified port")
        return False

def list_available_models():
    """List all available vLLM models and verify model IDs from servers"""
    print("\n" + "="*80)
    print("AVAILABLE vLLM MODELS")
    print("="*80)
    for key, model_info in AVAILABLE_MODELS.items():
        print(f"  [{key}] {model_info['description']}")
        print(f"      URL: {model_info['base_url']}")
        print(f"      Configured model name: {model_info['model_name']}")
        # Try to get actual model ID from server
        actual_id = get_model_id_from_server(model_info['base_url'])
        if actual_id:
            print(f"      Actual model ID from server: {actual_id}")
            if actual_id != model_info['model_name']:
                print(f"      ⚠ WARNING: Model name mismatch! Update config to use: {actual_id}")
        else:
            print(f"      ⚠ Could not connect to server to verify model ID")
    print("="*80 + "\n")

def select_model(model_key: str = None) -> Dict:
    """
    Select a model to use. If model_key is provided, use it. Otherwise, prompt user.
    
    Args:
        model_key: Optional model key (1, 2, 3, or 4). If None, prompts user.
    
    Returns:
        Dictionary with model configuration (base_url, model_name, etc.)
    """
    if model_key is None:
        # Interactive selection
        list_available_models()
        while True:
            choice = input("Select model (1-4) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Exiting...")
                sys.exit(0)
            if choice in AVAILABLE_MODELS:
                selected = AVAILABLE_MODELS[choice]
                print(f"\nSelected: {selected['description']}")
                return selected
            else:
                print(f"Invalid choice. Please enter 1, 2, 3, 4, or 'q'.")
    else:
        # Use provided key
        if model_key in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_key]
        else:
            print(f"Error: Invalid model key '{model_key}'. Available keys: {list(AVAILABLE_MODELS.keys())}")
            list_available_models()
            return select_model()  # Prompt user

async def classify_real_comments_vllm_async(comments_by_sector: Dict[str, List[str]], 
                                           output_prefix: str = "real_comments_vllm", 
                                           base_url: str = VLLM_BASE_URL,
                                           model_name: str = VLLM_MODEL_NAME,
                                           max_concurrent: int = MAX_CONCURRENT_REQUESTS):
    """
    Classify real Reddit comments by sector using survey questions with vLLM (async version).
    Optimized for continuous batching - maximum throughput.
    
    Args:
        comments_by_sector: Dictionary with sector as key and list of comment strings as values
                           e.g., {"Food": ["comment1", "comment2"], "Housing": ["comment3"]}
        output_prefix: Prefix for output JSON files (default: "real_comments_vllm")
        base_url: Base URL for the vLLM API (default: VLLM_BASE_URL)
        model_name: Model name as served by vLLM (default: VLLM_MODEL_NAME)
        max_concurrent: Maximum concurrent requests (default: MAX_CONCURRENT_REQUESTS)
    
    Returns:
        Dictionary with results organized by sector
    """
    print("="*80)
    print("CLASSIFYING REAL REDDIT COMMENTS BY SECTOR (vLLM)")
    print("="*80 + "\n")
    
    # Load survey questions organized by sector
    survey_by_sector = load_survey_questions_by_sector()
    if not survey_by_sector:
        print("Error: Could not load survey questions. Exiting.")
        return {}
    
    # Normalize sector names in input
    sector_mapping = {
        'food': 'Food',
        'transport': 'Transport',
        'housing': 'Housing',
        'energy': 'Energy'  # Legacy support, but main sectors are Food, Transport, Housing
    }
    
    print(f"Model: {model_name} | URL: {base_url} | Max concurrent: {max_concurrent}")
    print(f"Comments by sector: {', '.join([f'{s}: {len(c)}' for s, c in comments_by_sector.items()])}\n")
    
    # Store all results by sector
    all_sector_results = {}
    
    # Process each sector
    for sector, comments in comments_by_sector.items():
        # Normalize sector name
        normalized_sector = sector_mapping.get(sector.lower(), sector.capitalize())
        
        # Get questions for this sector ONLY (no cross-sector questions)
        sector_questions = get_questions_for_sector(normalized_sector)
        
        if not sector_questions:
            print(f"Warning: No questions found for sector '{sector}'. Skipping.")
            survey_by_sector = load_survey_questions_by_sector()
            print(f"Available sectors: {', '.join(survey_by_sector.keys())}")
            continue
        
        print(f"\n[{normalized_sector}] {len(comments)} comments, {len(sector_questions)} questions")
        
        # ========== FIRST PASS: RELEVANCE CHECK ==========
        
        start_time = time.time()
        relevance_results, first_pass_tokens = await first_pass_relevance_vllm_async(
            comments, sector_questions, 
            base_url=base_url, model_name=model_name, 
            max_concurrent=max_concurrent
        )
        first_pass_time = time.time() - start_time
        
        # Print relevance percentages
        all_relevance_results = {model_name: relevance_results}
        print_relevance_percentages(all_relevance_results, sector_questions, comments,
                                   has_ground_truth=False)
        
        # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
        
        start_time = time.time()
        classification_results, second_pass_tokens = await second_pass_classification_vllm_async(
            comments, sector_questions, relevance_results,
            base_url=base_url, model_name=model_name, 
            max_concurrent=max_concurrent
        )
        second_pass_time = time.time() - start_time
        
        # Count relevant pairs for throughput calculation
        relevant_pairs = sum(1 for stmt_idx in relevance_results 
                           for q_id in relevance_results[stmt_idx] 
                           if relevance_results[stmt_idx][q_id] == 'yes')
        
        # Store final results (using same structure as sample_local_llm.py for compatibility)
        all_results = {model_name: classification_results}
        
        # Save results to JSON (comments as keys, all model labels grouped)
        output_filename = f"{output_prefix}_{normalized_sector.lower()}_results.json"
        save_results_json(all_results, comments, sector_questions, filename=output_filename)
        
        # Save relevant-only results JSON
        relevant_only_filename = f"{output_prefix}_{normalized_sector.lower()}_results_relevant_only.json"
        save_results_json_relevant_only(all_results, comments, sector_questions, relevance_results, 
                                       filename=relevant_only_filename)
        
        # Calculate and print metrics (no ground truth)
        metrics = calculate_metrics(all_results, comments, sector_questions, has_ground_truth=False)
        print_metrics(metrics)
        
        # Store results for this sector
        all_sector_results[normalized_sector] = {
            'results': all_results,
            'metrics': metrics,
            'comments': comments,
            'questions': sector_questions,
            'first_pass_time': first_pass_time,
            'second_pass_time': second_pass_time,
            'first_pass_tokens': first_pass_tokens,
            'second_pass_tokens': second_pass_tokens,
            'relevant_pairs': relevant_pairs,
            'total_tasks': len(comments) * len(sector_questions)
        }
    
    # Print consolidated summary at the end
    print("\n" + "="*80)
    print("CLASSIFICATION COMPLETE FOR ALL SECTORS (vLLM)")
    print("="*80)
    total_time_all = sum(s['first_pass_time'] + s['second_pass_time'] for s in all_sector_results.values())
    total_tokens_all = sum(s['first_pass_tokens']['total_tokens'] + s['second_pass_tokens']['total_tokens'] for s in all_sector_results.values())
    total_output_tokens_all = sum(s['first_pass_tokens']['completion_tokens'] + s['second_pass_tokens']['completion_tokens'] for s in all_sector_results.values())
    total_requests_all = sum(s['total_tasks'] + s['relevant_pairs'] for s in all_sector_results.values())
    
    if total_time_all > 0:
        print(f"Summary: {total_requests_all} requests in {total_time_all:.1f}s | "
              f"{total_requests_all/total_time_all:.1f} req/s | "
              f"{total_tokens_all:,} tokens ({total_output_tokens_all/total_time_all:.0f} tok/s)")
    print("="*80 + "\n")
    
    return all_sector_results

def classify_real_comments_vllm(comments_by_sector: Dict[str, List[str]], 
                                output_prefix: str = "real_comments_vllm", 
                                base_url: str = VLLM_BASE_URL,
                                model_name: str = VLLM_MODEL_NAME,
                                max_concurrent: int = MAX_CONCURRENT_REQUESTS):
    """
    Synchronous wrapper for classify_real_comments_vllm_async
    Classify real Reddit comments by sector using survey questions with vLLM.
    
    Works in both regular Python scripts and Jupyter notebooks.
    In notebooks, use await classify_real_comments_vllm_async() directly for better performance.
    
    Args:
        comments_by_sector: Dictionary with sector as key and list of comment strings as values
                           e.g., {"Food": ["comment1", "comment2"], "Housing": ["comment3"]}
        output_prefix: Prefix for output JSON files (default: "real_comments_vllm")
        base_url: Base URL for the vLLM API (default: VLLM_BASE_URL)
        model_name: Model name as served by vLLM (default: VLLM_MODEL_NAME)
        max_concurrent: Maximum concurrent requests (default: MAX_CONCURRENT_REQUESTS)
    
    Returns:
        Dictionary with results organized by sector
    """
    try:
        # Check if we're in a running event loop (e.g., Jupyter notebook)
        loop = asyncio.get_running_loop()
        # If we're in a notebook, try to use nest_asyncio if available
        try:
            import nest_asyncio
            nest_asyncio.apply()
            # Now we can use asyncio.run() even in a running loop
            return asyncio.run(classify_real_comments_vllm_async(
                comments_by_sector, output_prefix, base_url, model_name, max_concurrent
            ))
        except ImportError:
            # nest_asyncio not available, provide helpful error
            raise RuntimeError(
                "Cannot use classify_real_comments_vllm() in a running event loop (Jupyter notebook).\n"
                "Option 1: Install nest_asyncio and apply it:\n"
                "  pip install nest_asyncio\n"
                "  import nest_asyncio\n"
                "  nest_asyncio.apply()\n"
                "  results = classify_real_comments_vllm(sample_comments)\n\n"
                "Option 2: Use the async version directly:\n"
                "  results = await classify_real_comments_vllm_async(sample_comments)"
            )
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or "get_running_loop" in str(e):
            # This is the specific error we're trying to handle
            raise
        # No running loop, safe to use asyncio.run()
        return asyncio.run(classify_real_comments_vllm_async(
            comments_by_sector, output_prefix, base_url, model_name, max_concurrent
        ))

async def main_async(model_config: Dict = None, use_real_survey_questions: bool = False):
    """Async main function to run the two-pass classification with vLLM (optimized for continuous batching)"""
    global SURVEY_QUESTIONS
    
    # Get model configuration
    if model_config is None:
        model_config = {"base_url": VLLM_BASE_URL, "model_name": VLLM_MODEL_NAME}
    
    base_url = model_config["base_url"]
    model_name = model_config["model_name"]
    
    # Reload questions if needed
    if use_real_survey_questions:
        SURVEY_QUESTIONS = load_survey_questions(use_real_survey_questions=True)
    
    print("="*80)
    print("vLLM SERVER SURVEY QUESTION CLASSIFIER (TWO-PASS APPROACH)")
    print("OPTIMIZED FOR CONTINUOUS BATCHING - MAXIMUM THROUGHPUT")
    print("="*80)
    if use_real_survey_questions:
        print(f"Using real survey data from survey_question.json")
    else:
        print(f"Survey: Pew Research - Americans' views on local wind and solar power development")
        print(f"URL: https://www.pewresearch.org/science/2024/06/27/americans-views-on-local-wind-and-solar-power-development/")
    print("="*80 + "\n")
    
    # Test connection to vLLM server
    if not test_vllm_connection(base_url):
        return
    
    print(f"\nProcessing with vLLM model: {model_name}")
    print(f"Server URL: {base_url}")
    print(f"Total questions: {len(SURVEY_QUESTIONS)}")
    print(f"Original statements: {len(SAMPLE_STATEMENTS)}")
    print(f"Max concurrent requests: {MAX_CONCURRENT_REQUESTS}")
    print(f"vLLM will automatically batch concurrent requests for optimal throughput\n")
    
    # Resample SAMPLE_STATEMENTS from 10 to 1000 for scaling test
    TARGET_SAMPLE_SIZE = 10
    original_count = len(SAMPLE_STATEMENTS)
    
    # Create mapping from resampled index to original index for ground truth lookup
    resampled_to_original = {}
    
    if original_count < TARGET_SAMPLE_SIZE:
        # Resample with replacement to reach target
        resampled_statements = []
        for i in range(TARGET_SAMPLE_SIZE):
            original_idx = random.randint(0, original_count - 1)
            resampled_statements.append(SAMPLE_STATEMENTS[original_idx])
            resampled_to_original[i] = original_idx
        print(f"Resampling statements: {original_count} -> {TARGET_SAMPLE_SIZE} (with replacement)")
        print(f"  Note: Some statements will appear multiple times to test scaling\n")
    else:
        # Sample without replacement if we have enough
        original_indices = list(range(original_count))
        sampled_indices = random.sample(original_indices, TARGET_SAMPLE_SIZE)
        resampled_statements = [SAMPLE_STATEMENTS[idx] for idx in sampled_indices]
        resampled_to_original = {i: idx for i, idx in enumerate(sampled_indices)}
        print(f"Sampling statements: {original_count} -> {TARGET_SAMPLE_SIZE} (without replacement)\n")
    
    # Store relevance results from first pass
    all_relevance_results = {}
    
    # ========== FIRST PASS: RELEVANCE CHECK ==========
    
    start_time = time.time()
    relevance_results, first_pass_tokens = await first_pass_relevance_vllm_async(
        resampled_statements, SURVEY_QUESTIONS, 
        base_url=base_url, model_name=model_name, 
        max_concurrent=MAX_CONCURRENT_REQUESTS
    )
    first_pass_time = time.time() - start_time
    all_relevance_results[model_name] = relevance_results
    
    total_first_pass_tasks = len(resampled_statements) * len(SURVEY_QUESTIONS)
    
    # Print relevance percentages
    print_relevance_percentages(all_relevance_results, SURVEY_QUESTIONS, resampled_statements,
                               has_ground_truth=not use_real_survey_questions, 
                               statement_to_original_index=resampled_to_original if not use_real_survey_questions else None)
    
    # ========== SECOND PASS: DETAILED CLASSIFICATION ==========
    
    # Second pass: Detailed classification only for relevant statements
    start_time = time.time()
    classification_results, second_pass_tokens = await second_pass_classification_vllm_async(
        resampled_statements, SURVEY_QUESTIONS, relevance_results,
        base_url=base_url, model_name=model_name, 
        max_concurrent=MAX_CONCURRENT_REQUESTS
    )
    second_pass_time = time.time() - start_time
    
    # Count relevant pairs for throughput calculation
    relevant_pairs = sum(1 for stmt_idx in relevance_results 
                       for q_id in relevance_results[stmt_idx] 
                       if relevance_results[stmt_idx][q_id] == 'yes')
    
    # Store final results (using same structure as sample_local_llm.py for compatibility)
    all_results = {model_name: classification_results}
    
    # Save results to CSV (using resampled statements)
    # Pass mapping so ground truth can be looked up correctly for resampled statements
    save_results_csv(all_results, resampled_statements, SURVEY_QUESTIONS, 
                    filename="vllm_test_results.csv", has_ground_truth=not use_real_survey_questions,
                    statement_to_original_index=resampled_to_original if not use_real_survey_questions else None)
    
    # Save relevant-only results CSV
    save_results_csv_relevant_only(all_results, resampled_statements, SURVEY_QUESTIONS, relevance_results,
                                  filename="vllm_test_results_relevant_only.csv", has_ground_truth=not use_real_survey_questions,
                                  statement_to_original_index=resampled_to_original if not use_real_survey_questions else None)
    
    # Calculate and print metrics (using resampled statements)
    # Pass mapping so ground truth can be looked up correctly for resampled statements
    metrics = calculate_metrics(all_results, resampled_statements, SURVEY_QUESTIONS, 
                               has_ground_truth=not use_real_survey_questions,
                               statement_to_original_index=resampled_to_original if not use_real_survey_questions else None)
    print_metrics(metrics)
    
    # Save metrics to CSV
    save_metrics_csv(metrics, filename="vllm_test_metrics.csv")
    
    total_time = first_pass_time + second_pass_time
    total_tokens = first_pass_tokens['total_tokens'] + second_pass_tokens['total_tokens']
    total_output_tokens = first_pass_tokens['completion_tokens'] + second_pass_tokens['completion_tokens']
    total_requests = total_first_pass_tasks + relevant_pairs
    
    print(f"\nSummary: {total_requests} requests in {total_time:.1f}s | "
          f"{total_requests/total_time:.1f} req/s | "
          f"{total_tokens:,} tokens ({total_output_tokens/total_time:.0f} tok/s)")
    
    print("Classification complete with vLLM (continuous batching optimized)!")

def main():
    """Main function wrapper (runs async version)"""
    # Check for model selection argument
    model_key = None
    if len(sys.argv) > 1:
        # Check if first arg is a model number (1-4)
        if sys.argv[1] in AVAILABLE_MODELS:
            model_key = sys.argv[1]
        # Check for --model argument
        elif len(sys.argv) > 2 and sys.argv[1] == "--model":
            model_key = sys.argv[2]
    
    # Select model (interactive if no key provided)
    model_config = select_model(model_key)
    
    # Update global defaults for this run
    global VLLM_BASE_URL, VLLM_MODEL_NAME
    VLLM_BASE_URL = model_config["base_url"]
    VLLM_MODEL_NAME = model_config["model_name"]
    
    asyncio.run(main_async(model_config))

def process_reddit_data_and_classify(model_config: Dict = None):
    """
    Process Reddit data from CSV files and classify comments by sector using vLLM.
    This function loads the filtered CSV files, maps keywords to sectors, and runs classification.
    
    Args:
        model_config: Optional model configuration dict. If None, uses global defaults.
    """
    if model_config is None:
        model_config = {"base_url": VLLM_BASE_URL, "model_name": VLLM_MODEL_NAME}
    print("="*80)
    print("PROCESSING REDDIT DATA AND CLASSIFYING WITH vLLM")
    print("="*80 + "\n")
    
    # ========== CHECK CACHE FIRST ==========
    cache_file = os.path.join('paper4data', 'sector_to_comments_cache.json')
    comments_by_sector = {}
    
    if os.path.exists(cache_file):
        print(f"Loading cached sector_to_comments from {cache_file}...")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                comments_by_sector = json.load(f)
            print(f"Loaded cached data:")
            for sector, comments in comments_by_sector.items():
                print(f"  {sector}: {len(comments)} comments")
            # Skip to sampling step if cache loaded successfully
            print("\nStep 5: Sampling comments per sector...")
            sample_comments = {}
            for sector, comments in comments_by_sector.items():
                n_samples = min(100, len(comments))
                sampled_comments = random.sample(comments, n_samples) if len(comments) > n_samples else comments
                sample_comments[sector] = sampled_comments
            
            print(f"Sampled comments by sector:")
            for sector, comments in sample_comments.items():
                print(f"  {sector}: {len(comments)} comments")
            
            # Skip to classification step
            print("\n" + "="*80)
            print("STEP 6: CLASSIFYING WITH vLLM")
            print("="*80)
            print("This will:")
            print("  - Food comments -> only Food survey questions")
            print("  - Transport comments -> only Transport survey questions")
            print("  - Housing comments -> only Housing survey questions")
            print("  NO CROSS-SECTOR CLASSIFICATION")
            print("  Uses vLLM server with continuous batching for maximum throughput")
            print("="*80 + "\n")
            
            # Test connection first
            if not test_vllm_connection(model_config["base_url"]):
                print("Cannot proceed without vLLM server connection")
                return None
            
            # Run classification
            results = asyncio.run(classify_real_comments_vllm_async(
                sample_comments,
                base_url=model_config["base_url"],
                model_name=model_config["model_name"]
            ))
            
            return results
        except Exception as e:
            print(f"Error loading cache file: {e}")
            print("Rebuilding from CSV files...")
            comments_by_sector = {}
    
    # ========== STEP 1: Define sector keywords ==========
    print("Step 1: Defining sector keywords...")
    sector_keyword_strength = {
        'transport_strong': [
            "electric vehicle", "evs", "bev", "battery electric", "battery-electric vehicle",
            "tesla model", "model 3", "model y", "chevy bolt", "nissan leaf",
            "ioniq 5", "mustang mach-e", "id.4", "rivian", "lucid air",
            "supercharger", "gigafactory", "zero emission vehicle", "zero-emission vehicle",
            "pure electric", "all-electric", "fully electric", "100% electric",
            "electric powertrain", "electric drivetrain", "electric motor vehicle",
            "level 2 charger", "dc fast charger", "public charger", "home charger",
            "charging network", "range anxiety", "mpge",
            "bike lane", "protected cycleway", "car-free", "low emission zone"
        ],
        'transport_weak': [
            "electric car", "electric truck", "electric suv", "plug-in hybrid",
            "phev", "charging station", "charge point", "kw charger", 
            "battery swap", "solid-state battery", "gigacast",
            "tax credit", "zev mandate", "ev rebate", "phase-out ice",
            "e-bike", "micro-mobility", "last-mile delivery", "transit electrification",
            "tesla", "spacex launch price?", "elon says",
            "rail electrification", "hydrogen truck", "low carbon transport"
        ],
        'housing_strong': [
            "rooftop solar", "solar pv", "pv panel", "photovoltaics",
            "solar array", "net metering", "feed-in tariff", "solar inverter",
            "kwh generated", "solar roof", "sunrun", "sunpower",
            r"solar\s+panel(s)?", r"solar\s+pv", r"rooftop\s+solar",
            r"solar\s+power", r"photovoltaic(s)?"
        ],
        'housing_weak': [
            "solar panels", "solar power", "solar installer",
            "battery storage", "powerwall", "home battery", "smart thermostat",
            "energy audit", "energy efficiency upgrade", "led retrofit",
            "green home", "net-zero house", "zero-energy building",
            "solar tax credit", "pvgis", "renewable portfolio standard",
            "community solar", "virtual power plant", "rooftop rebate"
        ],
        'food_strong': [
            "vegan", "plant-based diet", "veganism", "veganuary", "vegetarian", "veg lifestyle",
            "carnivore diet", "meat lover", "steakhouse", "barbecue festival",
            "bacon double", "grass-fed beef", "factory farming",
            "meatless monday", "beyond meat", "impossible burger",
            "plant-based burger", "animal cruelty free"
        ],
        'food_weak': [
            "red meat", "beef consumption", "dairy free", "plant protein",
            "soy burger", "nutritional yeast", "seitan", "tofurky",
            "agricultural emissions", "methane footprint", "carbon hoofprint",
            "cow burps", "livestock emissions", "feedlot",
            "recipe vegan", "tofu scramble", "almond milk", "oat milk",
            "flexitarian", "climatetarian",
            "cultivated meat", "lab-grown meat", "precision fermentation"
        ]
    }
    
    # Combine all keywords for regex search
    sector_keywords = (
        sector_keyword_strength['transport_strong'] +
        sector_keyword_strength['transport_weak'] +
        sector_keyword_strength['housing_strong'] +
        sector_keyword_strength['housing_weak'] +
        sector_keyword_strength['food_strong'] +
        sector_keyword_strength['food_weak']
    )
    
    # ========== STEP 2: Load filtered CSV files ==========
    print("Step 2: Loading filtered CSV files...")
    output_dir = os.path.join('paper4data', 'subreddit_filtered_by_regex')
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found. Please run the filtering step first.")
        return None
    
    filtered_csv_paths = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith('.csv')
    ]
    
    if not filtered_csv_paths:
        print(f"Error: No CSV files found in '{output_dir}'")
        return None
    
    print(f"Found {len(filtered_csv_paths)} CSV files")
    
    # Load all filtered CSVs
    all_filtered = []
    for path in filtered_csv_paths:
        try:
            df = pd.read_csv(path)
            all_filtered.append(df)
        except Exception as e:
            print(f"Error loading filtered csv {path}: {e}")
    
    if not all_filtered:
        print("Error: No data loaded from CSV files")
        return None
    
    df_all_filtered = pd.concat(all_filtered, ignore_index=True)
    print(f"Loaded {len(df_all_filtered)} total rows")
    
    # Fill empty/null body with title
    if 'body' in df_all_filtered.columns and 'title' in df_all_filtered.columns:
        mask = df_all_filtered['body'].isna() | (df_all_filtered['body'] == '') | (df_all_filtered['body'].astype(str).str.strip() == '')
        count_to_fill = mask.sum()
        df_all_filtered.loc[mask, 'body'] = df_all_filtered.loc[mask, 'title']
        if count_to_fill > 0:
            print(f"Filled {count_to_fill} empty/null body values with title")
    
    # Reconstruct keyword_to_comments
    keyword_to_comments = {}
    if not df_all_filtered.empty:
        for kw, group in df_all_filtered.groupby('matched_keyword'):
            keyword_to_comments[kw] = set(group['id'].astype(str))
    
    # Print overall stats
    if not df_all_filtered.empty:
        keyword_counts = df_all_filtered.groupby('matched_keyword')['id'].nunique().sort_values(ascending=False)
        print("\nTotal unique comments/submissions matched per keyword:")
        for kw, count in keyword_counts.head(10).items():  # Show top 10
            print(f"  {kw}: {count}")
        print(f"  ... and {len(keyword_counts) - 10} more keywords")
    
    # ========== STEP 3: Create sector mapping ==========
    print("\nStep 3: Creating sector mapping from matched_keyword...")
    
    # Create reverse mapping: keyword -> sector
    keyword_to_sector = {}
    
    # Map transport keywords
    for kw in sector_keyword_strength['transport_strong'] + sector_keyword_strength['transport_weak']:
        keyword_to_sector[kw] = 'transport'
    
    # Map housing keywords
    for kw in sector_keyword_strength['housing_strong'] + sector_keyword_strength['housing_weak']:
        keyword_to_sector[kw] = 'housing'
    
    # Map food keywords
    for kw in sector_keyword_strength['food_strong'] + sector_keyword_strength['food_weak']:
        keyword_to_sector[kw] = 'food'
    
    # Add sector column to df_all_filtered based on matched_keyword
    if not df_all_filtered.empty and 'matched_keyword' in df_all_filtered.columns:
        df_all_filtered['sector'] = df_all_filtered['matched_keyword'].map(keyword_to_sector)
        print(f"Added 'sector' column to df_all_filtered")
        print(f"Sector distribution:")
        print(df_all_filtered['sector'].value_counts())
    else:
        print("Warning: df_all_filtered is empty or missing 'matched_keyword' column")
        return None
    
    # ========== STEP 4: Prepare comments by sector ==========
    print("\nStep 4: Preparing comments by sector...")
    
    # Build comments_by_sector from filtered data (cache check already done at start)
    if not comments_by_sector:
        print("Building sector_to_comments from filtered data...")
        for sector in df_all_filtered['sector'].dropna().unique():
            sector_comments = df_all_filtered[df_all_filtered['sector'] == sector]['body'].dropna().tolist()
            # Filter out empty strings
            sector_comments = [c for c in sector_comments if c and str(c).strip()]
            if sector_comments:
                comments_by_sector[sector] = sector_comments
        
        # Save to cache
        print(f"Saving sector_to_comments to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(comments_by_sector, f, ensure_ascii=False, indent=2)
            print(f"Cache saved successfully")
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
        
        print(f"Comments by sector:")
        for sector, comments in comments_by_sector.items():
            print(f"  {sector}: {len(comments)} comments")
    
    # ========== STEP 5: Sample comments per sector ==========
    print("\nStep 5: Sampling comments per sector...")
    sample_comments = {}
    for sector, comments in comments_by_sector.items():
        n_samples = min(100, len(comments))
        sampled_comments = random.sample(comments, n_samples) if len(comments) > n_samples else comments
        sample_comments[sector] = sampled_comments
    
    print(f"Sampled comments by sector:")
    for sector, comments in sample_comments.items():
        print(f"  {sector}: {len(comments)} comments")
    
    # ========== STEP 6: Classify with vLLM ==========
    print("\n" + "="*80)
    print("STEP 6: CLASSIFYING WITH vLLM")
    print("="*80)
    print("This will:")
    print("  - Food comments -> only Food survey questions")
    print("  - Transport comments -> only Transport survey questions")
    print("  - Housing comments -> only Housing survey questions")
    print("  NO CROSS-SECTOR CLASSIFICATION")
    print("  Uses vLLM server with continuous batching for maximum throughput")
    print("="*80 + "\n")
    
    # Test connection first
    if not test_vllm_connection(model_config["base_url"]):
        print("Cannot proceed without vLLM server connection")
        return None
    
    # Run classification
    results = asyncio.run(classify_real_comments_vllm_async(
        sample_comments,
        base_url=model_config["base_url"],
        model_name=model_config["model_name"]
    ))
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    args = sys.argv[1:]
    model_key = None
    process_reddit = False
    use_real_survey_questions = False
    
    # Check for help
    if "--help" in args or "-h" in args:
        print("\n" + "="*80)
        print("vLLM TEST - USAGE")
        print("="*80)
        print("\nUsage:")
        print("  python vllm_test.py [model_number] [options]")
        print("\nModel Selection:")
        print("  [1] Qwen2.5-3B-Instruct (port 8000)")
        print("  [2] Qwen3 1.7B (port 8001)")
        print("  [3] Gemma 2 2B Instruct (port 8002)")
        print("  [4] Mistral MiniStral 3 3B (port 8003)")
        print("\nOptions:")
        print("  --process-reddit    Process Reddit data from CSV files")
        print("  --use-survey-questions  Use real survey questions from survey_question.json")
        print("  --model N           Select model by number (1-4)")
        print("  --help, -h          Show this help message")
        print("\nExamples:")
        print("  python vllm_test.py                                    # Interactive model selection")
        print("  python vllm_test.py 1                                  # Use model 1 (Qwen2.5-3B)")
        print("  python vllm_test.py 1 --process-reddit --use-survey-questions  # Process Reddit data with real survey questions")
        print("  python vllm_test.py --model 3                          # Use model 3 (Gemma 2)")
        print("="*80 + "\n")
        sys.exit(0)
    
    # Check for model selection
    if args and args[0] in AVAILABLE_MODELS:
        model_key = args[0]
        args = args[1:]
    elif len(args) > 1 and args[0] == "--model":
        model_key = args[1]
        args = args[2:]
    
    # Check for other flags
    if "--process-reddit" in args:
        process_reddit = True
    if "--use-survey-questions" in args:
        use_real_survey_questions = True
    
    # Select model (interactive if no key provided)
    model_config = select_model(model_key)
    
    if process_reddit:
        # Process Reddit data and classify
        results = process_reddit_data_and_classify(model_config)
        if results:
            print("\n" + "="*80)
            print("PROCESSING COMPLETE")
            print("="*80)
            print("Results saved to JSON files with prefix 'real_comments_vllm_'")
    else:
        # Default: run with sample statements
        asyncio.run(main_async(model_config, use_real_survey_questions))
