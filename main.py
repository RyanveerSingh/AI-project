import asyncio
from pipeline import run_pipeline


def main():
    print("=== AI Comparator ===")

    question = input("\nEnter your question: ")

    if not question.strip():
        print("No question entered.")
        return

    result = asyncio.run(run_pipeline(question))

    print("\n" + "=" * 50)
    print("MODEL A:", result["model_a_name"])
    print("=" * 50)
    print(result["answer_a"])

    print("\n" + "=" * 50)
    print("MODEL B:", result["model_b_name"])
    print("=" * 50)
    print(result["answer_b"])

    print("\n" + "=" * 50)
    print("JUDGE VERDICT")
    print("=" * 50)

    judgment = result["judgment"]

    if "error" in judgment:
        print("Judge Error:", judgment["error"])
        return

    print("Winner:", judgment.get("winner"))
    print("Score A:", judgment.get("score_a"))
    print("Score B:", judgment.get("score_b"))

    print("\n--- Model A Analysis ---")
    print("Strengths:", judgment.get("strengths_a"))
    print("Weaknesses:", judgment.get("weaknesses_a"))

    print("\n--- Model B Analysis ---")
    print("Strengths:", judgment.get("strengths_b"))
    print("Weaknesses:", judgment.get("weaknesses_b"))

    print("\n--- Final Combined Answer ---\n")
    print(judgment.get("final_answer"))


if __name__ == "__main__":
    main()