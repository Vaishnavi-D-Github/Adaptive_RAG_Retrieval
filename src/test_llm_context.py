import ollama


question = (
    "How many OT cybersecurity products and services did CISA "
    "provide to critical infrastructure owners and operators "
    "between October 2018 and October 2023?"
)


context = """
Appendix I: Objectives, Scope, and Methodology

The Cybersecurity and Infrastructure Security Agency reviewed
CISA's Industrial Control Systems Security Offerings, which lists
17 OT cybersecurity products and services.

We then removed one product—the Automated Indicator Sharing
System—that does not help critical infrastructure owners and
operators to address cyber OT risks.

We also removed two services—the Industrial Control System Joint
Working Group and technical analysis and one product—industrial
control system alerts—that CISA had retired.

We validated that the remaining 13 OT products and services were
offered to critical infrastructure owners and operators between
October 2018 and October 2023 by obtaining documentation and
written responses from CISA describing when these products and
services were offered.
"""


prompt = f"""
Answer the question using ONLY the information in the context.

QUESTION:
{question}

CONTEXT:
{context}

The context explicitly contains the answer.

What is the number of OT cybersecurity products and services?

Answer with the number and a short explanation.
"""


response = ollama.chat(
    model="llama3.2:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    options={
        "temperature": 0
    }
)


print("\nLLM RESPONSE:")
print(response["message"]["content"])