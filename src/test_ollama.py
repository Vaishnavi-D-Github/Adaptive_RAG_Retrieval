import ollama


response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": "In one sentence, explain what cybersecurity is."
        }
    ]
)


print("Ollama response:")
print(response["message"]["content"])