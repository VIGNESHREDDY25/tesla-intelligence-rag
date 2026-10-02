import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "In one sentence, explain what Tesla is."
        }
    ],
)

print(response.choices[0].message.content)