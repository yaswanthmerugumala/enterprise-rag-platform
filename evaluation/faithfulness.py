from core.llm.ollama_llm import OllamaLLM

judge_llm = OllamaLLM(model="phi3")  # lightweight model


def faithfulness_score(question, context, answer):

    prompt = f"""
You are an evaluator.

Question:
{question}

Context:
{context}

Answer:
{answer}

Does the answer strictly use the provided context?
Reply only with YES or NO.
"""

    response = judge_llm.generate(prompt)

    if "yes" in response.lower():
        return 1
    return 0
