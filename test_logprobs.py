"""
test_logprobs.py

Quick test to verify logprobs extraction from vLLM API works correctly.
Tests call_vllm_single_choice with a sample comment.
"""

import asyncio
import aiohttp
import json
from pathlib import Path

# Load the API config
with open("local_LLM_api_from_vLLM.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Use model key 6 (Mistral-7B on port 8001)
model_key = "6"
models = cfg["available_models"]
m = models[model_key]
base_url = m["base_url"]
model_name = m["model_name"]

print(f"Testing logprobs extraction with: {base_url} model={model_name}")

# Load norms schema
schema_file = Path(__file__).parent / "00_vllm_ipcc_social_norms_schema.json"
with open(schema_file, "r", encoding="utf-8") as f:
    schema = json.load(f)

NORMS_SYSTEM = schema["norms_system"]
test_question = schema["norms_questions"][0]  # First question (1.1_gate)

# Sample comment
test_comment = "I love electric vehicles! They're the future of transportation and will help reduce emissions."

async def test_logprobs():
    """Test logprobs extraction."""
    url = base_url.rstrip("/") + "/v1/chat/completions"

    prompt = test_question.get("prompt", "")
    user_content = prompt + "\n\n---\n\nComment/post:\n\n" + test_comment

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": NORMS_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
        "logprobs": True,
        "top_logprobs": 5,
    }

    async with aiohttp.ClientSession() as session:
        timeout = aiohttp.ClientTimeout(total=60)
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()

                print("\n" + "="*80)
                print("API Response")
                print("="*80)

                choice = (data.get("choices") or [{}])[0]
                content = choice.get("message", {}).get("content", "").strip()
                print(f"\nResponse content: {content}")

                # Check if logprobs are present
                if "logprobs" in choice:
                    logprobs_data = choice["logprobs"]
                    print(f"\n[OK] Logprobs present in response")
                    print(f"  Keys in logprobs: {list(logprobs_data.keys())}")

                    if "content" in logprobs_data and logprobs_data["content"]:
                        tokens = logprobs_data["content"]
                        print(f"  Number of tokens: {len(tokens)}")

                        # Show first few tokens
                        print(f"\n  First 3 tokens:")
                        for i, token_data in enumerate(tokens[:3]):
                            token = token_data.get("token", "?")
                            logprob = token_data.get("logprob")
                            print(f"    {i+1}. Token: '{token}' | Logprob: {logprob:.4f}")

                        # Calculate average logprob
                        token_logprobs = [t.get("logprob", 0) for t in tokens if t.get("logprob") is not None]
                        if token_logprobs:
                            avg_logprob = sum(token_logprobs) / len(token_logprobs)
                            print(f"\n  Average logprob: {avg_logprob:.4f}")
                            print(f"  Confidence (exp(avg_logprob)): {(100 * (2.718281828 ** avg_logprob)):.2f}%")
                        else:
                            print(f"\n  [FAIL] No valid logprobs found in tokens")
                    else:
                        print(f"  [FAIL] No 'content' field in logprobs")
                else:
                    print(f"\n[FAIL] No logprobs in response")
                    print(f"  Response keys: {list(choice.keys())}")

                print("\n" + "="*80)
                print("Test complete!")
                print("="*80)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

# Run test
asyncio.run(test_logprobs())
