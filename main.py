import os
from dotenv import load_dotenv  # Import load_dotenv
from huggingface_hub import InferenceClient, get_token
import google.generativeai as genai

# ================= CONFIGURATION =================
# Load environment variables from .env file first
load_dotenv()

# Get the key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Models to compare (Hugging Face)
MODEL_A = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_B = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Judge Model (Google Gemini)
JUDGE_MODEL = "gemini-2.5-flash" 
# Note: If 'gemini-2.5-flash' works for you specifically, change the line above to:
# JUDGE_MODEL = "gemini-2.5-flash"
# =================================================

def ask_hf_model(client, model_id, question):
    """Send a question to a Hugging Face model."""
    try:
        messages = [{"role": "user", "content": question}]
        response = client.chat_completion(model=model_id, messages=messages, max_tokens=1024)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def ask_gemini_judge(question, answer_a, answer_b):
    """Send both answers to Gemini with the detailed expert prompt."""
    try:
        # Configure with the loaded key
        if not GEMINI_API_KEY:
            return "Judge Error: API Key not found in .env file."
            
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(JUDGE_MODEL)
        
        prompt = f"""
You are an expert evaluator of AI responses. Your task is to compare two answers given by different models to the same question and determine which one is more accurate, complete, and reliable.

Follow these steps carefully:

1. Read the **question** and both **Model A** and **Model B** answers.
2. Evaluate each answer based on:
   * Factual accuracy
   * Completeness of explanation
   * Logical reasoning
   * Clarity and structure
   * Relevance to the question
3. Identify any **errors, missing information, or misleading statements** in both answers.
4. Decide which answer is **better overall**, and clearly explain why.
5. If both answers are partially correct or incomplete, use your **own knowledge** to correct them.
6. Provide a **final improved answer** that is the most accurate, clear, and complete version possible.

Output format:

Question:
{question}

Model A Answer:
{answer_a}

Model B Answer:
{answer_b}

Evaluation of Model A:
* Strengths:
* Weaknesses:

Evaluation of Model B:
* Strengths:
* Weaknesses:

Better Answer:
[Model A / Model B / Both need improvement]

Reason:
[Explain clearly]

Corrected & Improved Final Answer:
[Write the best possible answer using your own knowledge if necessary]
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Judge Error: {str(e)}"

def main():
    print("=== AI Comparator with Expert Gemini Judge ===")
    
    # Check Gemini Key properly
    if not GEMINI_API_KEY:
        print("\n[ERROR] GEMINI_API_KEY not found!")
        print("1. Ensure you have a file named '.env' in this folder.")
        print("2. Ensure the file contains: GEMINI_API_KEY=your_actual_key_here")
        print("3. Ensure there are no spaces around the '=' sign.")
        return

    # Check HF Token
    hf_token = get_token()
    if not hf_token:
        print("\n[ERROR] Hugging Face token not found.")
        print("Please run: hf auth login")
        return
    
    try:
        client = InferenceClient(token=hf_token)
    except Exception as e:
        print(f"[ERROR] Failed to initialize HF Client: {e}")
        return

    # 1. Get User Question
    question = input("\nEnter your question: ")
    if not question.strip():
        print("No question entered.")
        return

    # 2. Get Responses from HF Models
    print(f"\n--- Asking {MODEL_A.split('/')[-1]} ---")
    answer_a = ask_hf_model(client, MODEL_A, question)
    print(f"{MODEL_A.split('/')[-1]} responded.")

    print(f"\n--- Asking {MODEL_B.split('/')[-1]} ---")
    answer_b = ask_hf_model(client, MODEL_B, question)
    print(f"{MODEL_B.split('/')[-1]} responded.")

    # 3. Display Raw Responses (Truncated for readability)
    print("\n" + "="*40)
    print("=== RAW RESPONSES ===")
    print("="*40)
    # Show first 600 chars to see more context before truncation
    print(f"\n[Model A - {MODEL_A.split('/')[-1]}]:\n{answer_a[:600]}... (truncated)")
    print("-" * 30)
    print(f"\n[Model B - {MODEL_B.split('/')[-1]}]:\n{answer_b[:600]}... (truncated)")
    print("="*40)

    # 4. Ask Gemini to Judge
    print("\n--- Sending to Expert Gemini Judge ---")
    judgment = ask_gemini_judge(question, answer_a, answer_b)
    
    # 5. Display Verdict
    print("\n" + "="*50)
    print("=== EXPERT JUDGE'S VERDICT ===")
    print("="*50)
    print(judgment)
    print("="*50)

if __name__ == "__main__":
    main()