import asyncio
from providers import ask_hf_model_async, ask_gemini_judge, MODEL_A_ID, MODEL_B_ID

async def run_pipeline(question: str) -> dict:
    """Runs the full comparison pipeline."""
    print(f"🚀 Starting pipeline for: '{question}'")
    print("⏳ Querying models in parallel...")
    
    # Run both model queries in parallel
    task_a = ask_hf_model_async(MODEL_A_ID, question)
    task_b = ask_hf_model_async(MODEL_B_ID, question)
    
    answer_a, answer_b = await asyncio.gather(task_a, task_b)
    
    print("✅ Models responded. Sending to Judge...")
    
    # Get judgment
    judgment = await ask_gemini_judge(question, answer_a, answer_b)
    
    print("✅ Judge completed.")
    
    return {
        "model_a_name": "Llama-3-8B",
        "model_b_name": "Qwen2.5-Coder-32B",
        "answer_a": answer_a,
        "answer_b": answer_b,
        "judgment": judgment
    }
