"""
Example script showing how to use classify_real_comments function
to classify Reddit comments by sector using local LLMs
"""

from sample_local_llm import classify_real_comments, prepare_comments_from_dataframe
import pandas as pd

# ============================================================================
# METHOD 1: Direct dictionary input
# ============================================================================

# Example: Prepare your comments organized by sector
comments_by_sector = {
    "Food": [
        "I've been trying to reduce my meat consumption for environmental reasons.",
        "Vegan food is so expensive, I can't afford it.",
        "I love plant-based burgers, they taste great!",
        "Meatless Monday is a great way to start reducing meat intake."
    ],
    "Housing": [
        "I installed solar panels on my roof last year and my electricity bill dropped significantly.",
        "Solar panels are too expensive for most homeowners.",
        "Home battery storage makes solar power much more practical.",
        "I'm considering getting solar panels but worried about the upfront cost."
    ],
    "Transport": [
        "Electric vehicles are the future of transportation.",
        "I'm worried about EV charging infrastructure in my area.",
        "I'm considering buying an electric vehicle.",
        "Charging stations need to be more widespread."
    ]
}

# Call the classification function
results = classify_real_comments(
    comments_by_sector=comments_by_sector,
    output_prefix="reddit_comments",  # Output files will be: reddit_comments_food_results.csv, etc.
    base_url="http://127.0.0.1:1234"  # Your local LLM API endpoint
)

# ============================================================================
# METHOD 2: From pandas DataFrame (e.g., from df_all_filtered)
# ============================================================================

# If you have a DataFrame with 'body' and 'sector' columns:
# df_all_filtered = ...  # Your DataFrame
# 
# # Prepare comments from DataFrame
# comments_by_sector = prepare_comments_from_dataframe(
#     df_all_filtered,
#     text_column='body',    # Column with comment text
#     sector_column='sector' # Column with sector labels
# )
# 
# # Classify the comments
# results = classify_real_comments(
#     comments_by_sector=comments_by_sector,
#     output_prefix="reddit_comments",
#     base_url="http://127.0.0.1:1234"
# )

# ============================================================================
# Accessing Results
# ============================================================================

# The function returns a dictionary with results organized by sector
# You can access results like:
# results['Food']['results']      # Classification results for Food sector
# results['Food']['metrics']      # Metrics for Food sector (coherence, etc.)
# results['Food']['comments']      # Original comments
# results['Food']['questions']     # Questions used for classification

print("\nClassification complete!")
print(f"Processed {len(results)} sectors")

# Example: Print summary for each sector
for sector, sector_data in results.items():
    print(f"\n{sector}:")
    print(f"  Comments processed: {len(sector_data['comments'])}")
    print(f"  Questions evaluated: {len(sector_data['questions'])}")
    print(f"  Total coherence: {sector_data['metrics']['total_coherence']:.2f}%")
