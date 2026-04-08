# AI Model Comparator

This project compares answers from multiple AI models and uses a judge model to evaluate them.

## Architecture

User Question
↓
Model A (Llama)
Model B (Qwen)
↓
Judge Model (Gemini)
↓
Final Combined Answer

## Features

- Parallel model querying
- AI judge evaluation
- Strength and weakness analysis
- Final synthesized answer

## Run

```bash
pip install -r requirements.txt
python main.py
