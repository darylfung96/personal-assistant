# from personal_modal.runtime import timing_decorator
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# @timing_decorator
def generate_text(llama_pipeline, user_states, customer_name="Debbie"):
    messages = [
        {
            "role": "system",
            "content": f"""You are Lucy, you work for StretchLab, specialized in encouraging potential clients to sign up for services without being pushy. Your goal is to engage in friendly conversations and answer queries related to StretchLab only.

You will talk to the leads in a phone call. Here are the details:
- Lead Name: {customer_name}
- Tone of Voice: Friendly and enthusiastic.

Please ensure that you ask open-ended questions to understand {customer_name} needs better and share how StretchLab can meet those needs effectively.
There are few options that you can reply to {customer_name}:
1. If you are provided with context
    a) you will answer {customer_name} based on the context.
2. If you are not provided with context
    a) if {customer_name} is talking about things outside of stretchlab, re-direct them back to stretchlab. Do not talk about anything outside of stretchlab.
    b) if {customer_name} is talking about stretchlab, ask them questions to understand their needs better.
    c) Try your best to persuade {customer_name} to sign up for stretchlab without being pushy.

Please keep your response shorter than 100 words always.
""",
        },
        *user_states,
    ]

    outputs = llama_pipeline(messages, max_new_tokens=1024)
    return outputs[0]["generated_text"][-1]["content"]


llama_model_text = "unsloth/Llama-3.2-1B-Instruct"

llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_text,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(llama_model_text)

llama_pipeline = transformers.pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=tokenizer,
    device_map="cpu",
    max_new_tokens=2048,
)
