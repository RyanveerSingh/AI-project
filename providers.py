import os
import json
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient, get_token
import google.generativeai as genai

# Load env variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = get_token()

# Models
MODEL_A_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_B_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
JUDGE_MODEL_ID = "gemini-2.5-flash"

# HuggingFace client
hf_client = InferenceClient(token=HF_TOKEN)

# Gemini client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

    judge_model = genai.GenerativeModel(
        JUDGE_MODEL_ID,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json"
        }
    )
else:
    judge_model = None


# ================= PROMPT =================

JUDGE_PROMPT = """
You are an expert AI evaluator and reasoning assistant.

Your task is to analyze answers from multiple AI models.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Carefully evaluate both answers.

For EACH answer provide:
• strengths
• weaknesses
• accuracy assessment

Then decide which answer is better.

Finally:
Combine the best parts of BOTH answers and produce a final improved answer.
If both answers miss something important, add your own knowledge.

Return ONLY JSON in the following format:

{{
"score_a": <1-10>,
"score_b": <1-10>,
"winner": "A" or "B" or "Tie",
"strengths_a": "<analysis>",
"weaknesses_a": "<analysis>",
"strengths_b": "<analysis>",
"weaknesses_b": "<analysis>",
"final_answer": "<best combined answer>"
}}
"""


# ================= MODEL CALL =================

def ask_hf_model(model_id: str, question: str) -> str:
    try:
        messages = [{"role": "user", "content": question}]

        response = hf_client.chat_completion(
            model=model_id,
            messages=messages,
            max_tokens=700,
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


# ================= JUDGE =================

def ask_gemini_judge(question, answer_a, answer_b):

    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b
    )

    for attempt in range(3):

        response = judge_model.generate_content(prompt)

        raw = response.text

        try:
            return json.loads(raw)

        except:
            pass

        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return {"error": "Judge failed after retries"}