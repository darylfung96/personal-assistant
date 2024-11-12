import os

import modal

from personal_modal.image import hf_vol, image, load_models, whisper_vol
from personal_modal.llama import generate_text
from personal_modal.stt import stt

app = modal.App("personal-interviewer", image=image)
whisper_model_path = "/whisper"
huggingface_model_path = os.environ["HF_HOME"]


@app.cls(
    volumes={
        whisper_model_path: whisper_vol,
        huggingface_model_path: hf_vol,
    },
    gpu="A10G",
)
class TextModel:
    @modal.enter()
    def load(self):
        self.whisper_model, self.llama_pipeline, _ = load_models(
            whisper_model_path,
            None,
            "cuda",
        )

    @modal.method()
    def generate(self, audio_base64: str):
        text = stt(self.whisper_model, audio_base64)
        print(text)

        if text == "":
            return {"audio_base64": ""}

        # Prepare text for Llama model
        all_text = [
            text,
        ]

        print(" ".join(all_text))

        # Generate response
        response = generate_text(self.llama_pipeline, " ".join(all_text))
        response = " ".join(response.split(" "))

        return {"text": response}
