import os
import re
import json
import asyncio
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, get_token
import google.generativeai as genai

# Load environment variables
load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = get_token()

# Model IDs
MODEL_A_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_B_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
JUDGE_MODEL_ID = "gemini-2.5-flash"  # Faster flash model

# Initialize Clients
hf_client = InferenceClient(token=HF_TOKEN)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Configure generation for speed: limit tokens and lower temperature
    generation_config = genai.GenerationConfig(
        max_output_tokens=300,
        temperature=0.1,
        top_p=0.9,
        candidate_count=1
    )
    judge_model = genai.GenerativeModel(JUDGE_MODEL_ID, generation_config=generation_config)
else:
    judge_model = None

# ================= PROMPTS =================
# Optimized Prompt: Concise and strict JSON enforcement
JUDGE_PROMPT = """Analyze these answers to: "{question}"

Answer A: {answer_a}
Answer B: {answer_b}

Return ONLY valid JSON:
{{"score_a": int, "score_b": int, "winner": "A"|"B"|"Tie", "strengths_a": "str", "weaknesses_a": "str", "strengths_b": "str", "weaknesses_b": "str", "final_answer": "str"}}"""

def ask_hf_model(model_id: str, question: str) -> str:
    """Synchronous call to Hugging Face."""
    try:
        messages = [{"role": "user", "content": question}]
        response = hf_client.chat_completion(model=model_id, messages=messages, max_tokens=1024)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

async def ask_hf_model_async(model_id: str, question: str) -> str:
    """Async wrapper for Hugging Face calls."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask_hf_model, model_id, question)

async def ask_gemini_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Calls Gemini asynchronously with optimizations for speed."""
    if not judge_model:
        return {"error": "Gemini API Key missing"}

    try:
        # OPTIMIZATION 1: Truncate inputs to prevent slow processing of long texts
        # The judge rarely needs more than ~1000 chars to determine quality
        max_chars = 1000
        trunc_a = answer_a[:max_chars] + ("..." if len(answer_a) > max_chars else "")
        trunc_b = answer_b[:max_chars] + ("..." if len(answer_b) > max_chars else "")

        prompt = JUDGE_PROMPT.format(question=question, answer_a=trunc_a, answer_b=trunc_b)

        # Run synchronous Gemini call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: judge_model.generate_content(prompt))

        raw_text = response.text

        # Clean markdown if present
        clean_text = re.sub(r"```json|```", "", raw_text).strip()
        # Attempt to find JSON block if wrapped in text
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
            
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse failed: {str(e)}", "raw_response": raw_text if 'raw_text' in locals() else ""}
    except Exception as e:
        return {"error": f"Judge failed: {str(e)}", "raw_response": ""}
