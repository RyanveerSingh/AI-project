import asyncio
from pipeline import run_pipeline

def main():
    print("=== AI Comparator (Refactored v2) ===")
    question = input("\nEnter your question: ")
    
    if not question.strip():
        print("No question entered.")
        return

    # Run the async pipeline
    result = asyncio.run(run_pipeline(question))
    
    # Display Results
    print("\n" + "="*40)
    print("=== RESULTS ===")
    print("="*40)
    
    print(f"\n🤖 {result['model_a_name']}:\n{result['answer_a'][:200]}... (truncated)")
    print(f"\n🤖 {result['model_b_name']}:\n{result['answer_b'][:200]}... (truncated)")
    
    print("\n" + "-"*40)
    print("⚖️ JUDGE'S VERDICT (JSON):")
    print("-"*40)
    
    judgment = result['judgment']
    if "error" in judgment:
        print(f"❌ Error: {judgment['error']}")
    else:
        print(f"🏆 Winner: Model {judgment['winner']}")
        print(f"📊 Scores: A={judgment['score_a']}, B={judgment['score_b']}")
        print(f"💡 Final Answer: {judgment['final_answer']}")

if __name__ == "__main__":
    main()
