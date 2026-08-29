from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
    think=False,
)

prompt = """
You are evaluating whether an answer is supported by a context.

Return ONLY valid JSON.

Use exactly this format:
{"verdict": "yes", "score": 1.0}

Context:
Federal agencies rely on tools, services, and resources for cybersecurity incident response.

Answer:
Federal agencies rely on tools, services, and resources for cybersecurity incident response.
"""

response = model.invoke(prompt)

print("=" * 80)
print("QWEN3:8B RAW RESPONSE")
print("=" * 80)

print(repr(response.content))

print()
print("=" * 80)
print("QWEN3:8B RESPONSE")
print("=" * 80)

print(response.content)

print()
print("Response type:", type(response.content))