import random
import re

from personal_modal.runtime import timing_decorator


def insert_uh(text, insert_probability=0.2):
    words = text.split()
    new_words = []

    for word in words:
        new_words.append(word)
        if random.random() < insert_probability:
            new_words.append("uh...")

    return " ".join(new_words)


@timing_decorator
def generate_text(llama_pipeline, user_states, customer_name="Debbie"):
    messages = [
        {
            "role": "system",
            "content": f"""You are Lucy, a lead generator for StretchLab, specialized in encouraging potential clients to sign up for services without being pushy. 
            Your goal is to engage in friendly conversations and answer queries related to StretchLab only.

Here are some of the following details that you need:
- Lead Name: {customer_name}
- Key Benefits of StretchLab: Improve range of motion and flexibility of muscles.
- Services Offered: Stretching
- Call Duration Goal: less than 5 minutes.
- Tone of Voice: Friendly and enthusiastic.

Please ensure that you ask open-ended questions to understand their needs better and share how StretchLab can meet those needs effectively.
Please keep your response shorter than 100 words always.
""",
        },
        *user_states,
    ]

    outputs = llama_pipeline(messages, max_new_tokens=1024)

    response = outputs[0]["generated_text"][-1]["content"]
    response = re.sub(r"[^a-zA-Z0-9,.!? ]+", "", response)
    response = insert_uh(response, insert_probability=0.05)

    return response
