import ollama


MODEL = "llama3.2:latest"

question = "How many OT cybersecurity products and services did CISA provide?"

context = """
The report states that the remaining 13 OT cybersecurity products
and services were offered to critical infrastructure owners and
operators between October 2018 and October 2023.
"""

prompt = f"""
Answer the question using only the information in the context.

If the context does not contain sufficient evidence to answer the
question, respond exactly:

"The available documents do not contain enough information to answer this question."

Question:
{question}

Context:
{context}

Answer:
"""

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0,
        "num_predict": 256
    }
)

print("=" * 80)
print("OLLAMA RESPONSE")
print("=" * 80)

print(response)

print("\n" + "=" * 80)
print("TOKEN USAGE")
print("=" * 80)

print(
    "Prompt tokens:",
    response.get("prompt_eval_count")
)

print(
    "Generated tokens:",
    response.get("eval_count")
)

print(
    "Prompt evaluation duration:",
    response.get("prompt_eval_duration")
)

print(
    "Generation duration:",
    response.get("eval_duration")
)