from runtime import timing_decorator


@timing_decorator
def generate_text(llama_pipeline, text):
    messages = [
        {"role": "user", "content": text},
    ]

    outputs = llama_pipeline(messages, max_new_tokens=512)
    return outputs[0]["generated_text"][-1]["content"]
