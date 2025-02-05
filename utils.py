from openai import OpenAI
import dotenv

dotenv.load_dotenv(".env")

def chat(prompt, image=None):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.01,
    )
    return response.choices[0].message.content