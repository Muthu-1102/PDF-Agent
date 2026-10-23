import os
import dotenv
from groq import Groq

dotenv.load_dotenv("backend/.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = """You are the Evidence Agent in an Agentic RAG system.
Your only job is to judge retrieved passages -- you do NOT answer the
question yourself.

Question: Who is the CEO of Acme Corp?

Candidate passages:
[0] (test.pdf, page 1): Alice Smith was appointed CEO of Acme Corp in 2021.
[1] (test.pdf, page 2): Acme Corp reported revenue of 50M in 2022.

For each passage, decide if it is actually relevant and useful for
answering the question (not just topically similar). Then decide if the
relevant passages, taken together, are SUFFICIENT to answer the question
completely and accurately.

Respond with ONLY a JSON object, no other text, in this exact shape:
{
  "relevant_indices": [list of integer indices of relevant passages],
  "sufficient": true or false,
  "reasoning": "one sentence explaining the sufficiency decision"
}"""

for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
    try:
        res = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        print("MODEL:", m)
        print("RESPONSE:\n", res.choices[0].message.content.strip())
        print("-" * 50)
    except Exception as e:
        print("ERROR with", m, e)
