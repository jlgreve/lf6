from openai import OpenAI

client = OpenAI(
    api_key="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"
)


def get_gpt_response(message: str):
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8UZe4KeA",
        messages=[{
            "role": "user",
            "content": message
        }],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content
