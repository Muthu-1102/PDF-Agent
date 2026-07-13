from groq import Groq

from app.core.config import settings

_client = Groq(api_key=settings.GROQ_API_KEY)


def call_llm(prompt: str, temperature: float = 0.2) -> str:
    response = _client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content
