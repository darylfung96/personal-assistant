import base64
import os

from personal_modal.image import hf_vol, image, load_models, voice_vol, whisper_vol
from personal_modal.llama import generate_text
from personal_modal.stt import stt
from personal_modal.speech.tts import generate_audio

import modal

app = modal.App("personal-interviewer", image=image)
voice_model_path = "/voice"
whisper_model_path = "/whisper"
huggingface_model_path = os.environ["HF_HOME"]


@app.cls(
    volumes={
        voice_model_path: voice_vol,
        whisper_model_path: whisper_vol,
        huggingface_model_path: hf_vol,
    },
    gpu="A10G",
)
class Model:
    @modal.enter()
    def load(self):
        self.whisper_model, self.llama_pipeline, self.voice_model = load_models(
            whisper_model_path,
            voice_model_path,
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
            "You will receive a question. You will be in an interview. Please answer it like how a human would and limit to 50 words. ONLY 50 WORDS and please provide just pure text without any symbols.",
            text,
        ]

        print(" ".join(all_text))

        # Generate response
        response = generate_text(self.llama_pipeline, " ".join(all_text))
        response = " ".join(response.split(" ")[:50])

        # Generate audio from response
        audio, audio_filepath = generate_audio(self.voice_model, response)

        with open(audio_filepath, "rb") as f:
            audio_data = f.read()

        output_audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        return {"audio_base64": output_audio_base64}
