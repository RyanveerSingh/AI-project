import asyncio
from providers import ask_hf_model, MODEL_A_ID, MODEL_B_ID, ask_gemini_judge


async def ask_model_async(model_id: str, question: str):

    loop = asyncio.get_event_loop()

    return await loop.run_in_executor(
        None,
        ask_hf_model,
        model_id,
        question
    )


async def run_pipeline(question: str):

    print(f"🚀 Starting pipeline for: '{question}'")

    print("⏳ Querying models in parallel...")

    task_a = asyncio.create_task(
        ask_model_async(MODEL_A_ID, question)
    )

    task_b = asyncio.create_task(
        ask_model_async(MODEL_B_ID, question)
    )

    answer_a, answer_b = await asyncio.gather(task_a, task_b)

    print("✅ Models responded. Sending to Judge...")

    loop = asyncio.get_event_loop()

    judgment = await loop.run_in_executor(
        None,
        ask_gemini_judge,
        question,
        answer_a,
        answer_b
    )

    print("✅ Judgment received.")

    return {
        "question": question,
        "model_a_name": MODEL_A_ID.split("/")[-1],
        "answer_a": answer_a,
        "model_b_name": MODEL_B_ID.split("/")[-1],
        "answer_b": answer_b,
        "judgment": judgment
    }