# from personal_modal.runtime import timing_decorator
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# @timing_decorator
def generate_text(llama_pipeline, user_states, customer_name="Debbie"):
    messages = [
        {
            "role": "system",
            "content": f"""You are Lucy, a lead generator for StretchLab, specialized in encouraging potential clients to sign up for services without being pushy. Your goal is to engage in friendly conversations and answer queries related to StretchLab only.

Please generate a script for contacting leads. Here are the details:
- Lead Name: {customer_name}
- Key Benefits of StretchLab: Improve range of motion and flexibility of muscles.
- Services Offered: Stretching
- Call Duration Goal: less than 5 minutes.
- Tone of Voice: Friendly and enthusiastic.

Please ensure that you ask open-ended questions to understand their needs better and share how StretchLab can meet those needs effectively.
""",
        },
        *user_states,
    ]

    outputs = llama_pipeline(messages, max_new_tokens=1024)
    return outputs[0]["generated_text"][-1]["content"]


llama_model_text = "unsloth/Llama-3.2-1B-Instruct"

llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_text,
    load_in_4bit=True,
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
